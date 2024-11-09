# src/workflow.py
from typing import Dict, List
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from datetime import datetime
from pydantic import BaseModel

from nodes.receiver import TextPreprocessor, TextInput
from nodes.encoder import TextEncoder
from nodes.curator import CuratorAgent, FraudAnalysis
from nodes.utils import EmbeddingStorage, FraudTypeRegistry

# Define state types using Pydantic for better validation
class WorkflowState(BaseModel):
    """State model for the fraud detection workflow."""
    messages: List[BaseMessage]
    text_input: TextInput | None = None
    embeddings: Dict | None = None
    similar_cases: List[str] | None = None
    analysis: FraudAnalysis | None = None
    should_alert: bool = False

class FraudDetectionConfig(BaseModel):
    """Configuration for the fraud detection workflow."""
    similarity_threshold: float = 0.85
    min_similar_cases: int = 3
    confidence_threshold: float = 0.7

def create_workflow(
    preprocessor: TextPreprocessor,
    encoder: TextEncoder,
    curator: CuratorAgent,
    embedding_storage: EmbeddingStorage,
    type_registry: FraudTypeRegistry,
    config: FraudDetectionConfig = FraudDetectionConfig()
) -> StateGraph:
    """Create the fraud detection workflow graph with updated LangGraph syntax."""
    
    def preprocess(state: dict) -> dict:
        """Preprocess incoming text."""
        state = WorkflowState.model_validate(state)
        messages = state.messages
        last_message = messages[-1]
        
        # Process the text
        text_input = preprocessor.process_text(
            last_message.content,
            source="user_input",
            metadata={"timestamp": datetime.now().isoformat()}
        )
        
        return WorkflowState(
            messages=messages,
            text_input=text_input,
            should_alert=False
        ).model_dump()
    
    def encode(state: dict) -> dict:
        """Generate embeddings and find similar cases."""
        state = WorkflowState.model_validate(state)
        
        if not state.text_input:
            return state.model_dump()
            
        # Generate embedding
        current_embedding = encoder.encode_text(state.text_input)
        
        # Find similar cases using vectorized operations
        stored_embeddings = []
        stored_metadata = []
        
        # Batch load stored embeddings
        for stored_file in embedding_storage.storage_dir.glob("*.npy"):
            stored_embedding, metadata = embedding_storage.load_embedding(stored_file.stem)
            stored_embeddings.append(stored_embedding)
            stored_metadata.append(metadata)
            
        if stored_embeddings:
            # Vectorized similarity computation
            import numpy as np
            similarities = np.dot(stored_embeddings, current_embedding) / (
                np.linalg.norm(stored_embeddings, axis=1) * np.linalg.norm(current_embedding)
            )
            
            # Filter similar cases
            similar_indices = similarities > config.similarity_threshold
            similar_cases = [
                stored_metadata[i].get("text", "")
                for i in np.where(similar_indices)[0]
            ]
            
            similar_embeddings = [
                {
                    "embedding": stored_embeddings[i],
                    "similarity": float(similarities[i]),
                    "metadata": stored_metadata[i]
                }
                for i in np.where(similar_indices)[0]
            ]
        else:
            similar_cases = []
            similar_embeddings = []
        
        return WorkflowState(
            messages=state.messages,
            text_input=state.text_input,
            embeddings={
                "current": current_embedding,
                "similar": similar_embeddings
            },
            similar_cases=similar_cases,
            should_alert=False
        ).model_dump()
    
    def should_curate(state: dict) -> str:
        """Enhanced decision logic for curation."""
        state = WorkflowState.model_validate(state)
        
        if not state.text_input:
            return "end"
            
        similar_cases = state.similar_cases or []
        
        # More sophisticated decision logic
        if len(similar_cases) < config.min_similar_cases:
            return "curate"
            
        # Check confidence based on similarities
        if state.embeddings and state.embeddings.get("similar"):
            similarities = [case["similarity"] for case in state.embeddings["similar"]]
            avg_similarity = sum(similarities) / len(similarities)
            
            if avg_similarity > config.confidence_threshold:
                return "analyze_similar"
            
        return "curate"
    
    def curate(state: dict) -> dict:
        """Detailed fraud analysis with enhanced storage."""
        state = WorkflowState.model_validate(state)
        
        if not state.text_input:
            return state.model_dump()
            
        analysis = curator.analyze_case(
            state.text_input,
            state.similar_cases or [],
            type_registry.get_types()
        )
        
        # Store results if fraud detected
        if analysis.is_fraud and state.embeddings:
            # Generate unique identifier
            case_id = f"fraud_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Store embedding with enhanced metadata
            embedding_storage.save_embedding(
                state.embeddings["current"],
                {
                    "text": state.text_input.text,
                    "fraud_type": analysis.fraud_type,
                    "confidence": analysis.confidence,
                    "timestamp": datetime.now().isoformat(),
                    "similar_cases_count": len(state.similar_cases or []),
                    "case_id": case_id
                },
                case_id
            )
            
            # Register new fraud type if applicable
            if analysis.fraud_type == "NEW":
                type_registry.add_type(analysis.fraud_type)
        
        return WorkflowState(
            messages=state.messages,
            text_input=state.text_input,
            embeddings=state.embeddings,
            similar_cases=state.similar_cases,
            analysis=analysis,
            should_alert=analysis.is_fraud
        ).model_dump()
    
    def analyze_similar(state: dict) -> dict:
        """Quick analysis based on similar cases with enhanced logic."""
        state = WorkflowState.model_validate(state)
        similar_cases = state.similar_cases or []
        
        if not similar_cases:
            return state.model_dump()
        
        # Enhanced analysis logic
        if state.embeddings and state.embeddings.get("similar"):
            similarities = [case["similarity"] for case in state.embeddings["similar"]]
            avg_similarity = sum(similarities) / len(similarities)
            
            # Determine fraud type based on similar cases
            fraud_types = [
                case["metadata"].get("fraud_type")
                for case in state.embeddings["similar"]
                if case["metadata"].get("fraud_type")
            ]
            
            most_common_type = max(set(fraud_types), key=fraud_types.count) if fraud_types else "SIMILAR_TO_KNOWN"
            
            analysis = FraudAnalysis(
                is_fraud=True,
                confidence=avg_similarity,
                fraud_type=most_common_type,
                explanation=f"Similar to {len(similar_cases)} known fraud cases of type {most_common_type}",
                similar_cases=similar_cases,
                timestamp=datetime.now()
            )
        else:
            analysis = FraudAnalysis(
                is_fraud=False,
                confidence=0.0,
                fraud_type=None,
                explanation="Insufficient similarity data",
                similar_cases=similar_cases,
                timestamp=datetime.now()
            )
        
        return WorkflowState(
            messages=state.messages,
            text_input=state.text_input,
            embeddings=state.embeddings,
            similar_cases=similar_cases,
            analysis=analysis,
            should_alert=analysis.is_fraud
        ).model_dump()
    
    # Create graph with new syntax
    workflow = StateGraph(WorkflowState)
    
    # Add nodes with error handling
    for node_name, node_fn in [
        ("preprocess", preprocess),
        ("encode", encode),
        ("curate", curate),
        ("analyze_similar", analyze_similar)
    ]:
        workflow.add_node(node_name, node_fn)
    
    # Add edges with new conditional syntax
    workflow.add_edge("preprocess", "encode")
    workflow.add_conditional_edges(
        "encode",
        should_curate,
        {
            "curate": "curate",
            "analyze_similar": "analyze_similar",
            "end": END
        }
    )
    workflow.add_edge("curate", END)
    workflow.add_edge("analyze_similar", END)
    
    workflow.set_entry_point("preprocess")
    
    return workflow.compile()

def run_fraud_detection(
    text: str,
    config: FraudDetectionConfig = FraudDetectionConfig()
) -> Dict:
    """Run the fraud detection workflow with configuration."""
    try:
        # Initialize components
        preprocessor = TextPreprocessor()
        encoder = TextEncoder()
        curator = CuratorAgent()
        embedding_storage = EmbeddingStorage()
        type_registry = FraudTypeRegistry()
        
        # Create workflow
        workflow = create_workflow(
            preprocessor,
            encoder,
            curator,
            embedding_storage,
            type_registry,
            config
        )
        
        # Prepare initial state
        initial_state = WorkflowState(
            messages=[HumanMessage(content=text)],
            should_alert=False
        ).model_dump()
        
        # Run workflow
        final_state = workflow.invoke(initial_state)
        final_state = WorkflowState.model_validate(final_state)
        
        # Return results
        return {
            "is_fraud": final_state.analysis.is_fraud if final_state.analysis else False,
            "confidence": final_state.analysis.confidence if final_state.analysis else 0.0,
            "fraud_type": final_state.analysis.fraud_type if final_state.analysis else None,
            "explanation": final_state.analysis.explanation if final_state.analysis else "",
            "should_alert": final_state.should_alert,
            "similar_cases_count": len(final_state.similar_cases or [])
        }
        
    except Exception as e:
        print(f"Error in fraud detection: {e}")
        return {
            "error": str(e),
            "is_fraud": False,
            "confidence": 0.0,
            "fraud_type": None,
            "explanation": "Error processing case",
            "should_alert": False,
            "similar_cases_count": 0
        }