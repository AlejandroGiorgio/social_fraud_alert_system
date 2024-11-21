# src/workflow.py
from typing import Dict, List, Union, Optional
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
import json
import numpy as np

from src.nodes.receiver import TextPreprocessor, TextInput
from src.nodes.encoder import TextEncoder
from src.nodes.curator_graph import CuratorAgent, FraudAnalysis, TextInput as CuratorTextInput
from src.nodes.utils import EmbeddingStorage, FraudTypeRegistry

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy arrays and scalars"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()  # Convert to Python native types
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        return super().default(obj)

class WorkflowState(BaseModel):
    """State model for the fraud detection workflow."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={
            np.ndarray: lambda x: x.tolist(),
            np.integer: int,
            np.floating: float
        }
    )
    
    messages: List[BaseMessage]
    text_input: Optional[TextInput] = None
    embeddings: Optional[Dict] = None
    similar_cases: Optional[List[str]] = None
    analysis: Optional[FraudAnalysis] = None
    should_alert: bool = False
    new_type_name: Optional[str] = None

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
    """Create the fraud detection workflow graph."""
    
    def preprocess(state: dict) -> dict:
        """Preprocess incoming text."""
        try:
            state = WorkflowState.model_validate(state)
            messages = state.messages
            last_message = messages[-1]

            clean_text = last_message.content.replace('\n', ' ').replace('\r', '')

            clean_text = clean_text.encode('ascii', 'ignore').decode('ascii')
            
            # Process the text
            text_input = preprocessor.process_text(
                clean_text,
                source="user_input",
                metadata={"timestamp": datetime.now().isoformat()}
            )
            
            return WorkflowState(
                messages=messages,
                text_input=text_input,
                should_alert=False
            ).model_dump()
        except Exception as e:
            print(f"Preprocessing error: {e}")
            raise

    def encode(state: dict) -> dict:
        """Generate embeddings and find similar cases."""
        try:
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
                # Convert current_embedding to list to avoid serialization issues
                current_embedding_list = current_embedding.tolist()

                # Compute similarities
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
                        "embedding": stored_embeddings[i].tolist(),  # Convert to list
                        "similarity": float(similarities[i]),  # Ensure float
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
                    "current": current_embedding.tolist(),  # Convert to list
                    "similar": similar_embeddings
                },
                similar_cases=similar_cases,
                should_alert=False
            ).model_dump()
        except Exception as e:
            print(f"Encoding error: {e}")
            raise

    def curate(state: dict) -> dict:
        """Detailed fraud analysis with enhanced storage."""
        try:
            state = WorkflowState.model_validate(state)
            
            if not state.text_input:
                return state.model_dump()
            
            # Convert to CuratorTextInput
            curator_input = CuratorTextInput(text=state.text_input.text)
            
            # Use existing CuratorAgent's analyze_case method
            analysis = curator.analyze_case(
                curator_input,
                state.similar_cases or []
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
                        "timestamp": datetime.now().isoformat(),
                        "similar_cases_count": len(state.similar_cases or []),
                        "case_id": case_id
                    },
                    case_id
                )
            
            return WorkflowState(
                messages=state.messages,
                text_input=state.text_input,
                embeddings=state.embeddings,
                similar_cases=state.similar_cases,
                analysis=analysis,
                should_alert=analysis.is_fraud,
                new_type_name=analysis.new_type_name
            ).model_dump()
        except Exception as e:
            print(f"Curation error: {e}")
            raise

    # Create graph
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("preprocess", preprocess)
    workflow.add_node("encode", encode)
    workflow.add_node("curate", curate)
    
    # Add edges
    workflow.add_edge("preprocess", "encode")
    workflow.add_edge("encode", "curate")
    workflow.add_edge("curate", END)
    
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
        
       # Preparar estado inicial
        initial_state = WorkflowState(
            messages=[HumanMessage(content=text)],
            should_alert=False
        )
        
        # Ejecutar workflow
        final_state = workflow.invoke(initial_state.model_dump())
        final_state = WorkflowState.model_validate(final_state)

        new_type_name = None

        if final_state.new_type_name:
            type_registry.add_type(final_state.new_type_name)
            new_type_name = final_state.new_type_name

        # Resto del código igual
        return {
            "is_fraud": final_state.analysis.is_fraud if final_state.analysis else False,
            "fraud_type": final_state.analysis.fraud_type if final_state.analysis else None,
            "explanation": final_state.analysis.explanation if final_state.analysis else "",
            "should_alert": final_state.should_alert,
            "new_type_name": new_type_name,
            "similar_cases_count": len(final_state.similar_cases or [])
        }
        
    except Exception as e:
        print(f"Error in fraud detection: {e}")
        return {
            "error": str(e),
            "is_fraud": False,
            "fraud_type": None,
            "explanation": "Error processing case",
            "should_alert": False,
            "similar_cases_count": 0
        }