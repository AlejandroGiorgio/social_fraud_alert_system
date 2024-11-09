from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime
from langchain.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.llms import Ollama
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
import json

class FraudTypeRegistry:
    """Registry for known fraud types."""
    
    def __init__(self):
        self._types: Set[str] = {
            "PHISHING",
            "IDENTITY_THEFT",
            "FINANCIAL_SCAM",
            "IMPERSONATION",
            "DATA_BREACH"
        }
    
    def add_type(self, fraud_type: str) -> None:
        """Add a new fraud type to the registry."""
        self._types.add(fraud_type.upper())
    
    def get_types(self) -> List[str]:
        """Get list of known fraud types."""
        return sorted(list(self._types))
    
    def has_type(self, fraud_type: str) -> bool:
        """Check if a fraud type exists in the registry."""
        return fraud_type.upper() in self._types

# State models
class CuratorState(BaseModel):
    """State for the curator subgraph"""
    messages: List[BaseMessage]
    text: str
    similar_cases: List[str]
    pattern_analysis: Optional[Dict] = None
    fraud_type: Optional[Dict] = None
    final_summary: Optional[str] = None
    is_fraud: bool = False

class TextInput(BaseModel):
    """Schema for text input."""
    text: str = Field(..., description="The text to analyze")

# Tool input models
class FraudPatternInput(BaseModel):
    """Schema for fraud pattern analysis."""
    text: str = Field(..., description="The text to analyze for fraud patterns")
    similar_cases: List[str] = Field(..., description="List of similar cases for reference")

class FraudTypeInput(BaseModel):
    """Schema for fraud type classification."""
    description: str = Field(..., description="Description of the fraud pattern")
    known_types: List[str] = Field(..., description="List of known fraud types")

class CuratorTools:
    """Tools for fraud analysis, separated from flow control"""
    
    def __init__(self, llm: Ollama):
        self.llm = llm
    
    @tool
    def analyze_fraud_patterns(self, input: FraudPatternInput) -> Dict:
        """Analyze text for fraud patterns by comparing with similar cases."""
        prompt = f"""Analyze this text for potential fraud patterns:
        Text: {input.text}
        
        Compare with these similar cases:
        {json.dumps(input.similar_cases, indent=2)}
        
        Provide analysis in JSON format with these fields:
        - is_fraud: boolean indicating if this is likely a fraud
        - patterns: list of identified fraud patterns
        - reasoning: brief explanation of the analysis
        """
        
        response = self.llm.invoke(prompt)
        return json.loads(response)
    
    @tool
    def classify_fraud_type(self, input: FraudTypeInput) -> Dict:
        """Classify the type of fraud based on description and known types."""
        prompt = f"""Classify this fraud pattern:
        Description: {input.description}
        
        Known fraud types: {', '.join(input.known_types)}
        
        Return JSON with:
        - fraud_type: either one of the known types or "NEW" if it's a new pattern
        - explanation: brief explanation of classification
        - new_type_name: suggested name if it's a new pattern
        """
        
        response = self.llm.invoke(prompt)
        return json.loads(response)
    
    @tool
    def generate_fraud_summary(self, analysis: Dict) -> str:
        """Generate a clear summary of the fraud analysis for alerts."""
        prompt = f"""Generate a clear, concise summary of this fraud analysis:
        {json.dumps(analysis, indent=2)}
        
        Focus on:
        1. Type of fraud
        2. Key warning signs
        3. Recommended precautions
        
        Keep it brief but informative for public alerts.
        """
        
        return self.llm.invoke(prompt)

def create_curator_subgraph(
    llm: Ollama,
    type_registry: FraudTypeRegistry
) -> StateGraph:
    """Create the curator subgraph with LangGraph."""
    
    tools = CuratorTools(llm)
    
    def analyze_patterns(state: Dict) -> Dict:
        """First node: Analyze patterns and determine if fraud."""
        state = CuratorState.model_validate(state)
        
        analysis = tools.analyze_fraud_patterns(
            FraudPatternInput(
                text=state.text,
                similar_cases=state.similar_cases
            )
        )
        
        return CuratorState(
            messages=state.messages,
            text=state.text,
            similar_cases=state.similar_cases,
            pattern_analysis=analysis,
            is_fraud=analysis["is_fraud"]
        ).model_dump()
    
    def should_classify(state: Dict) -> str:
        """Decision node: Determine if we should proceed with classification."""
        state = CuratorState.model_validate(state)
        return "classify" if state.is_fraud else "end"
    
    def classify_type(state: Dict) -> Dict:
        """Second node: Classify fraud type if fraud was detected."""
        state = CuratorState.model_validate(state)
        
        type_result = tools.classify_fraud_type(
            FraudTypeInput(
                description=state.pattern_analysis["reasoning"],
                known_types=type_registry.get_types()
            )
        )
        
        return CuratorState(
            messages=state.messages,
            text=state.text,
            similar_cases=state.similar_cases,
            pattern_analysis=state.pattern_analysis,
            fraud_type=type_result,
            is_fraud=state.is_fraud
        ).model_dump()
    
    def generate_summary(state: Dict) -> Dict:
        """Final node: Generate summary for fraud cases."""
        state = CuratorState.model_validate(state)
        
        summary = tools.generate_fraud_summary({
            "pattern_analysis": state.pattern_analysis,
            "classification": state.fraud_type
        })
        
        return CuratorState(
            messages=state.messages,
            text=state.text,
            similar_cases=state.similar_cases,
            pattern_analysis=state.pattern_analysis,
            fraud_type=state.fraud_type,
            final_summary=summary,
            is_fraud=state.is_fraud
        ).model_dump()
    
    # Create graph
    workflow = StateGraph(CuratorState)
    
    # Add nodes
    workflow.add_node("analyze_patterns", analyze_patterns)
    workflow.add_node("classify_type", classify_type)
    workflow.add_node("generate_summary", generate_summary)
    
    # Add edges with conditional routing
    workflow.add_conditional_edges(
        "analyze_patterns",
        should_classify,
        {
            "classify": "classify_type",
            "end": END
        }
    )
    workflow.add_edge("classify_type", "generate_summary")
    workflow.add_edge("generate_summary", END)
    
    workflow.set_entry_point("analyze_patterns")
    
    return workflow.compile()

@dataclass
class FraudAnalysis:
    """Results of fraud analysis."""
    is_fraud: bool
    fraud_type: Optional[str]
    explanation: str
    similar_cases: List[str]
    timestamp: datetime

class CuratorAgent:
    """Refactored CuratorAgent that uses LangGraph subgraph."""
    
    def __init__(self, model_name: str = "mistral"):
        self.llm = Ollama(model=model_name)
        self.type_registry = FraudTypeRegistry()
        self.subgraph = create_curator_subgraph(self.llm, self.type_registry)
    
    def analyze_case(
        self,
        text_input: TextInput,
        similar_cases: List[str]
    ) -> FraudAnalysis:
        """Analyze a case using the curator subgraph."""
        
        # Prepare initial state
        initial_state = CuratorState(
            messages=[],  # We don't need messages for the subgraph
            text=text_input.text,
            similar_cases=similar_cases
        ).model_dump()
        
        # Run subgraph
        final_state = self.subgraph.invoke(initial_state)
        final_state = CuratorState.model_validate(final_state)
        
        # Process results
        if final_state.is_fraud:
            return FraudAnalysis(
                is_fraud=True,
                fraud_type=final_state.fraud_type["fraud_type"] if final_state.fraud_type else None,
                explanation=final_state.final_summary or final_state.pattern_analysis["reasoning"],
                similar_cases=similar_cases,
                timestamp=datetime.now()
            )
        
        return FraudAnalysis(
            is_fraud=False,
            fraud_type=None,
            explanation=final_state.pattern_analysis["reasoning"],
            similar_cases=similar_cases,
            timestamp=datetime.now()
        )