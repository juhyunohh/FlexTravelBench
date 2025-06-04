import json
import logging
import pandas as pd
import random
import os
import re
from datetime import datetime
from pathlib import Path

# import dataclass
from dataclasses import dataclass
from typing import Any, Dict, Optional

import yaml
from openai import OpenAI
from together import Together
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from agents.swarm import Agent, Result, Swarm
from agents.utils import get_json_prompt
from agents.constraints_checker import planning_validate_constraints
from agents.prompts import CONSTRAINT_ADDING_W_HISTORY, CONSTRAINT_ADDING_WO_HISTORY, INITIAL_PROMPT
from agents.constraints_generator import update_constraints_and_query
from agents.postprocess_plan import parse_plan


@dataclass
class Response:
    messages: list
    agent: Agent
    context_variables: dict

class Evaluator(Agent):
    def __init__(
        self,
        config: dict,
        name: Optional[str] = None,
    ):

        evaluator_config = config["evaluator"]
        name = name or evaluator_config["name"]
        client_kwargs = evaluator_config.get("client", None)
        client = OpenAI(**client_kwargs) if client_kwargs else OpenAI()
        task = config['task']
        instructions = ""
        super().__init__(name=name, instructions=instructions, client=client)
        self.task = task
        self.evaluator_config = evaluator_config
        # self.constraints_dict = self.inital_constraints
        
    def check_constraint(self, constraints: str, response: str):
        if self.task == "test":
            current_constraint_score, total_score = planning_validate_constraints(
                turn_constraint=constraints,
                updated_constraints=self.constraints_dict, 
                response_data=response,
                ref_data=self.ref_data)
            # calculate the pass rate of the constraints (is_correct==True/total length of constraints)
            correct_count = sum(1 for item in total_score.values() if item.get('is_correct') is True)
            # total_count = len(current_constraint_score)
            # total count is where the constraints 'is_correct' is not None
            total_count = sum(1 for item in total_score.values() if item.get('query_value') is not None)
            if total_count == 0:
                print("No constraints to evaluate")
                return current_constraint_score, total_score, 0
            total_pass_rate = correct_count / total_count
        return current_constraint_score, total_score, total_pass_rate

    def check_quality(self, response) -> Result:
        # TODO: Implement quality check. Commonsense constraints, etc.
        final_quality_scores = "None"
        return final_quality_scores
    
    def conclude_evaluate(self, score: int, comments: str) -> Result:
        """End interview with final assessment.

        Called when max questions reached, understanding established, or unable to progress further.
        Also called when forced to conclude interview.

        Args:
            score (int): Final score (0-10) based on rubric
            comments (str): Overall evaluation including strengths,
                weaknesses, and areas for improvement

        Returns:
            Result: Final assessment with score and detailed feedback
        """
        return Result(
            value=f"Interview concluded. Score: {score}\nComments: {comments}",
            context_variables={
                "interview_complete": True,
                "score": score,
                "comments": comments,
            },
        )


class Evaluatee(Agent):
    def __init__(
        self,
        config: dict,
        name: Optional[str] = None,
    ):

        evaluatee_config = config["evaluatee"]
        name = name or evaluatee_config["name"]
        model = evaluatee_config["model"]
        api_type = evaluatee_config.get("api_type", "openai")
        client_kwargs = evaluatee_config.get("client", None)
        if api_type == 'together':
            client = Together(api_key=evaluatee_config.get("client", {}).get("api_key"))
        else:
            client = OpenAI(**client_kwargs) if client_kwargs else OpenAI()

        instructions = evaluatee_config["instructions"]
        super().__init__(name=name, model=model,instructions=instructions, client=client, api_type=api_type)

class Runner:
    def __init__(
        self,
        evaluator: Agent,
        evaluatee: Agent,
        config: dict,
        logger: logging.Logger,
        log_file_path: str,
        console: Console,
    ):
        self.client = Swarm()
        self.evaluator = evaluator
        self.evaluatee = evaluatee
        self.config = config
        self.logger = logger
        self.log_file_path = log_file_path
        self.console = console
        self.questions_count = 0
        self.max_constraints = len(self.evaluator.new_constraints)
        self.constraints = self.evaluator.new_constraints
        self.evaluator_messages = []
        self.evaluatee_messages = []
        self.scores = {"constraints":[]}
        self.all_total_scores = []  # Add this line
        self.questions = []
        self.responses = []
        self.seed_question_used = False
        # self.constraints_dict = self.evaluator.inital_constraints
        self.task = self.evaluator.task
        


    def display_message(self, agent_name: str, content: str):
        """Display a message with proper formatting."""

        agent_name_to_style = {
            self.evaluator.name.lower(): "blue",
            self.evaluatee.name.lower(): "green",
            "constraint checker": "yellow",
            "overall quality checker": "magenta",
        }

        style = agent_name_to_style[agent_name.lower()]
        panel = Panel(
            content,
            title=f"[{style}]{agent_name}[/{style}]",
            border_style=style,
            padding=(1, 2),
        )
        # Only print to console if in verbose mode
        if self.logger.getEffectiveLevel() <= logging.INFO:
            self.console.print(panel)

        # Always log to file if file logging is enabled
        self.logger.info(f"{agent_name}: {content}")

    def display_results(self, results: Dict[str, Any]):
        """Display interview results with formatting."""
        score = results["score"]
        score_color = "success" if score >= 0.7 else "warning" if score >= 0.5 else "error"

        results_panel = Panel(
            f"\n[{score_color}]Final Score: {score}[/{score_color}]\n\n"
            f"[info]Constrained Added: {results['questions_asked']}[/info]\n\n"
            f"[white]Constraint Count:[/white]\n{results['constraint_counts']}",
            title="[success]Assessment Results[/success]",
            border_style="success",
            padding=(1, 2),
        )
        self.console.print("\n")
        self.console.print(results_panel)

    def _get_response(self, agent: Agent, messages: list,
                      context: dict) -> Result:
        """Helper method to get response with progress spinner."""
        
        if not self.config['history']:
            messages = [messages[-1]]
        messages = [{"role": message["role"], "content": str(message["content"])} for message in messages]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        ) as progress:
            task = progress.add_task("Processing response...", total=None)
            return self.client.run(agent=agent, messages=messages, context_variables=context)

    def _get_response_raw(self, agent: Agent, messages: list,
                          chat_params: dict, json: bool = False) -> Response:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        ) as progress:
            task = progress.add_task("Processing response...", total=None)
            full_params = {
                "model": agent.model,
                "messages": messages,
            }

            if json:
                full_params["response_format"] = {"type": "json_object"}

            full_params.update(chat_params)
            raw_response = agent.client.chat.completions.create(**full_params)
            content = raw_response.choices[0].message.content
            return Response(messages=[
                            {"role": "assistant", "content": content}], agent=agent, context_variables={})

    def add_message(self, speaker, content):
        """Add messages to both conversation tracks based on who's speaking.

        When evaluator speaks: they're the assistant, evaluatee is the user
        When evaluatee speaks: they're the assistant, evaluator is the user
        """
        if speaker == self.evaluator:
            # evaluator is speaking (as assistant) to evaluatee (as user)
            self.evaluator_messages.extend(
                [{"role": "assistant", "content": content}])
            self.evaluatee_messages.extend(
                [{"role": "user", "content": content}])
        else:
            # evaluatee is speaking (as assistant) to evaluator (as user)
            self.evaluator_messages.extend(
                [{"role": "user", "content": content}])
            self.evaluatee_messages.extend(
                [{"role": "assistant", "content": content}])

    def call_feedback_agent(self, response):
        # constraints = self.evaluator.new_constraints[:self.questions_count]
        # if self.questions_count == 0, constraints = '' else, constraints = self.evaluator.new_constraints[self.questions_count]
        constraints = self.evaluator.new_constraints[self.questions_count - 1] if self.questions_count > 0 else ''
        current_score, total_score, total_pass_rate = self.evaluator.check_constraint(constraints,response)
        return current_score, total_score, total_pass_rate

    def call_question_agent(self):
        if self.questions_count == 0:
            query_to_display = self.evaluator.seed_question
            response = INITIAL_PROMPT.format(question = self.evaluator.seed_question, ref_data=self.evaluator.ref_data)
        else:
            constraint_seed = self.evaluator.new_constraints[self.questions_count-1]
            previous_condition = "\n".join(f"{key}: {value}" for key, value in self.evaluator.constraints_dict.items())
            self.evaluator.constraints_dict, constraint = update_constraints_and_query(self.evaluator.constraints_dict, constraint_seed)
            query_to_display = constraint
            
            if self.config['history']:
                response = CONSTRAINT_ADDING_W_HISTORY.format(constraint=constraint)
            else:
                response = CONSTRAINT_ADDING_WO_HISTORY.format(
                    ref_data = self.evaluator.ref_data,
                    previous_condition=previous_condition, 
                    additional_condition=constraint,
                    response=self.responses[-1],
                )
        return query_to_display, response


    # Note: Please follow the convention of adding the message to the
    # conversation first and then displaying it
    def run(self) -> Dict[str, Any]:
        """Run the evaluation and return results."""
        # Start the evaluation loop
        
        self.console.print(f"\n[info]Constraints {self.questions_count}[/info]")
        
        self.seed_question_used = True
        query_to_display, response = self.call_question_agent()
        
        self.questions.append(response)
        self.add_message(self.evaluator, response)
        self.display_message(self.evaluator.name,
                             query_to_display)

        while self.questions_count <= (self.max_constraints):
            # 1. Get response from evaluatee
            question = self.evaluator_messages[-1]["content"]

            response = self._get_response(
                self.evaluatee, self.evaluatee_messages, {}).messages[-1]["content"]
            parsed_plan = parse_plan(response)
            if len(parsed_plan) == 0:
                self.console.print(f"\n[warning]No plan found in the response. Please provide a valid plan.[/warning]")
            
            # self.responses.append(response)
            self.responses.append(parsed_plan)
            self.add_message(self.evaluatee, parsed_plan)
            self.display_message(self.evaluatee.name, str(parsed_plan))
            
            # 2. Get feedback from feedback agent. Note that the message from
            # feedback agent is not added to the conversation
            # current_score, total_score = self.call_feedback_agent(response)
            current_score, total_score, total_pass_rate = self.call_feedback_agent(parsed_plan)

            self.scores['constraints'].append(current_score)
            self.scores['total_constraints_score'] = total_score
            self.scores['total_pass_rate'] = total_pass_rate
            self.all_total_scores.append(total_score)  # Add this line
            self.scores['all_total_constraints_score'] = self.all_total_scores

            self.display_message(
                "Constraint Checker",
                "Current_constraints_score: {}\nTotal_score: {}".format(
                    current_score, total_score
                ),
            )

            # Constraints
            self.questions_count += 1
            if self.questions_count <= self.max_constraints:
                self.console.print(f"\n[info]Constraints {self.questions_count}")
                query_to_display, response = self.call_question_agent()
                self.questions.append(response)
                self.add_message(self.evaluator, response)
                rm_txt = '[Reference information]:\nAll costs are per one person, one night.\n' + str(self.evaluator.ref_data) + '\n\n' 
                display_response = response.replace(rm_txt, '')
                self.display_message(self.evaluator.name, display_response)

            # 4. Check end conditions for the interview
            if self.questions_count == (self.max_constraints+1):
                final_message = "Maximum number of constraints reached. Concluding Evaluation."
                self.console.print(f"\n[warning]{final_message}[/warning]")

                # quality_score = self.evaluator.check_quality(response)
                # just use the final hard constraint pass rate as the quality score
                # quality_score = total_pass_rate
                self.scores['quality'] = total_pass_rate
                panel_content = f"Total Pass Rate: {total_pass_rate}"

                self.display_message("Overall Quality Checker",
                                     panel_content)
                break
            
        # print(self.scores)
        results = {
            # "score": response.context_variables["score"],
            # score: 그동안의 pass rate의 평균
            "constraint_counts": sum(1 for item in self.scores['total_constraints_score'].values() if item.get('is_correct') is not None),
            "score": self.scores['total_pass_rate'],
            # "feedback": response.context_variables["comments"],
            "questions_asked": self.questions_count - 1,
            "all_total_scores": self.all_total_scores,
        }

        self.display_results(results)
        return results


