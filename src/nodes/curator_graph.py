from typing import Dict, List, Optional, Any, Type
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langgraph.graph import StateGraph, END
import os
import json
from tenacity import retry, stop_after_attempt, wait_exponential
from src.nodes.utils import FraudTypeRegistry
from src.settings import Settings
import ast

setting = Settings()


class FraudAnalysis(BaseModel):
    """Data class to store fraud analysis results."""

    is_fraud: bool
    fraud_type: Optional[str]
    explanation: str
    similar_cases: List[str]
    timestamp: datetime


# Existing State and Input models
class CuratorState(BaseModel):
    """State model for the fraud detection workflow."""

    text: str
    similar_cases: List[str]
    pattern_analysis: Optional[Dict] = None
    fraud_type: Optional[Dict] = None
    final_summary: Optional[str] = None
    is_fraud: bool = False


class TextInput(BaseModel):
    text: str = Field(..., description="The text to analyze")


# New Pydantic models for validation
class PatternAnalysisOutput(BaseModel):
    is_fraud: bool
    patterns: List[str]
    reasoning: str


class FraudTypeOutput(BaseModel):
    fraud_type: str
    explanation: str
    new_type_name: Optional[str] = None


class FraudSummaryOutput(BaseModel):
    summary: str
    warning_signs: List[str]
    precautions: List[str]

class PostResponseInput(BaseModel):
    content: Dict[str, Any] = Field(..., description="The content to validate and format")
    model: Type[BaseModel] = Field(..., description="The Pydantic model to validate against")

def post_response(input_data: PostResponseInput) -> Dict:
    """Tool for agents to post their responses in the correct format."""
    return input_data.model.model_validate(input_data.content).model_dump()


# Agent base class with retry logic
class BaseAgent:
    output_model: Type[BaseModel] = None  # Cada hijo debe definir su modelo de salida
    
    def __init__(self, llm: ChatOpenAI):
        if not self.output_model:
            raise ValueError("output_model must be defined in child class")
            
        self.llm = llm
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "post_response",
                    "description": "Post a response in the correct format according to the specified model",
                    "parameters": self.output_model.model_json_schema()  # Usa el schema del modelo específico
                }
            }
        ]

    def _validate_and_parse_response(self, response: str) -> Any:
        """Now uses the child's specific output_model."""
        try:

             # Reemplazar comillas simples por dobles para cumplir con el estándar JSON
            if isinstance(response, str):
                response = response.replace("'", '"')

            response_dict = json.loads(response) if isinstance(response, str) else response

            return post_response(PostResponseInput(content=response_dict, model=self.output_model))
        except Exception as e:
            print(f"Validation error: {e}. Retrying...")
            raise e



# Specific agents for each task
class PatternAnalysisAgent(BaseAgent):

    output_model = PatternAnalysisOutput

    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    "You are a fraud detection expert. Analyze the provided text for potential fraud patterns. "
                    "First you must determine if the case is fraud. "
                    "If it is fraud, provide a brief reasoning and list of patterns detected. "
                    "Use the 'post_response' tool to format your response with the following structure: "
                    "'is_fraud': boolean, 'patterns': [string], 'reasoning': string"
                ),
                HumanMessagePromptTemplate.from_template(
                    """Analyze this text for potential fraud patterns:
                Text: {text}
                
                Similar cases for reference:
                {similar_cases}"""
                ),
            ]
        )

    def analyze(self, text: str, similar_cases: List[str]) -> PatternAnalysisOutput:
        response = self.llm.invoke(
            self.prompt.format_messages(text=text, similar_cases=similar_cases)
        )
        print("RESPONSE FROM PATTERNANALYSISAGENT", response)
        return self._validate_and_parse_response(
            response.content
        )


class FraudTypeAgent(BaseAgent):

    output_model = FraudTypeOutput

    def __init__(self, llm: ChatOpenAI, type_registry: "FraudTypeRegistry"):
        super().__init__(llm)
        self.type_registry = type_registry
        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    "You are a fraud classification expert. Classify the provided fraud pattern into known categories. "
                    "You must classify the pattern into known fraud types."
                    "If it's a new type, you must stablish the fraud type as 'NEW' and provide a suggested name."
                    "Use the 'post_response' tool to format your response with the following structure: "
                    "'fraud_type': string, 'explanation': string, 'new_type_name': string?"
                ),
                HumanMessagePromptTemplate.from_template(
                    """Classify this fraud pattern:
                Description: {description}
                
                Known fraud types: {known_types}"""
                ),
            ]
        )

    def classify(self, pattern_description: str) -> FraudTypeOutput:
        response = self.llm.invoke(
            self.prompt.format_messages(
                description=pattern_description,
                known_types=", ".join(self.type_registry.get_types()),
            )
        )
        print("RESPONSE FROM FRAUDTYPEAGENT", response)
        return self._validate_and_parse_response(response.content)


class SummaryAgent(BaseAgent):

    output_model = FraudSummaryOutput

    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    "You are a fraud prevention expert."
                    "Generate a clear, abstract, and concise summary of this fraud analysis."
                    "Also, provide brief warning signs and precautions to prevent similar frauds."
                    "Use the 'post_response' tool to format your response with the following structure: "
                    "'summary': string, 'warning_signs': [string], 'precautions': [string]"
                ),
                HumanMessagePromptTemplate.from_template(
                    """Generate a clear, concise summary of this fraud analysis:
                {analysis}"""
                ),
            ]
        )

    def summarize(self, analysis: Dict) -> FraudSummaryOutput:
        response = self.llm.invoke(self.prompt.format_messages(analysis=analysis))
        print("RESPONSE FROM SUMMARYAGENT", response)
        return self._validate_and_parse_response(response.content)


# Modified workflow creation function
def create_curator_graph(
    pattern_agent: PatternAnalysisAgent,
    type_agent: FraudTypeAgent,
    summary_agent: SummaryAgent,
) -> StateGraph:
    """Create the curator workflow graph with agents."""
    workflow = StateGraph(CuratorState)

    # Node functions that use the agents
    def analyze_patterns(state: Dict) -> Dict:
        state = CuratorState.model_validate(state)
        result = pattern_agent.analyze(state.text, state.similar_cases)
        try:
            result_model = PatternAnalysisOutput.model_validate(result)
        except Exception as e:
            raise ValueError(f"Error al validar el resultado de PatternAnalysisAgent: {e}")
        state.pattern_analysis = result_model.model_dump()  # Serializar para el estado
        state.is_fraud = result_model.is_fraud
        return state.model_dump()


    def classify_type(state: Dict) -> Dict:
        state = CuratorState.model_validate(state)
        result = type_agent.classify(state.pattern_analysis)
        try:
            result_model = FraudTypeOutput.model_validate(result)
        except Exception as e:
            raise ValueError(f"Error al validar el resultado de FraudTypeAgent: {e}")
        state.fraud_type = result_model.model_dump()
        return state.model_dump()


    def generate_summary(state: Dict) -> Dict:
        state = CuratorState.model_validate(state)
        result = summary_agent.summarize(
            {
                "pattern_analysis": state.pattern_analysis,
                "fraud_type": state.fraud_type,
            }
        )
        # Validar y convertir el resultado al modelo esperado
        try:
            result_model = FraudSummaryOutput.model_validate(result)
        except Exception as e:
            raise ValueError(f"Error al validar el resultado de SummaryAgent: {e}")
        state.final_summary = result_model.summary
        return state.model_dump()


    # Add nodes
    workflow.add_node("analyze_patterns", analyze_patterns)
    workflow.add_node("classify_type", classify_type)
    workflow.add_node("generate_summary", generate_summary)

    # Define conditional routing
    def should_classify(state: Dict) -> str:
        state = CuratorState.model_validate(state)
        return "classify" if state.is_fraud else "end"

    # Add edges
    workflow.add_conditional_edges(
        "analyze_patterns", should_classify, {"classify": "classify_type", "end": END}
    )
    workflow.add_edge("classify_type", "generate_summary")
    workflow.add_edge("generate_summary", END)

    workflow.set_entry_point("analyze_patterns")

    return workflow.compile()


# Modified CuratorAgent class
class CuratorAgent:
    """CuratorAgent that uses OpenAI for inference with validated agents"""

    def __init__(self):

        settings = Settings()

        self.llm = ChatOpenAI(model=settings.OPENAI_LLM_MODEL, temperature=0.0)

        self.type_registry = FraudTypeRegistry()

        # Initialize agents
        self.pattern_agent = PatternAnalysisAgent(self.llm)
        self.type_agent = FraudTypeAgent(self.llm, self.type_registry)
        self.summary_agent = SummaryAgent(self.llm)

        # Create workflow with agents
        self.workflow = create_curator_graph(
            self.pattern_agent, self.type_agent, self.summary_agent
        )

    def analyze_case(
        self, text_input: TextInput, similar_cases: List[str]
    ) -> FraudAnalysis:
        """Analyze a potential fraud case."""
        initial_state = CuratorState(
            text=text_input.text, similar_cases=similar_cases
        ).model_dump()

        final_state = self.workflow.invoke(initial_state)
        final_state = CuratorState.model_validate(final_state)

        if final_state.is_fraud:
            return FraudAnalysis(
                is_fraud=True,
                fraud_type=(
                    final_state.fraud_type["fraud_type"]
                    if final_state.fraud_type
                    else None
                ),
                explanation=final_state.final_summary
                or final_state.pattern_analysis["reasoning"],
                similar_cases=similar_cases,
                timestamp=datetime.now(),
            )

        return FraudAnalysis(
            is_fraud=False,
            fraud_type=None,
            explanation=final_state.pattern_analysis["reasoning"],
            similar_cases=similar_cases,
            timestamp=datetime.now(),
        )