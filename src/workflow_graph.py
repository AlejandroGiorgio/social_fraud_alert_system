from typing import Dict, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
import numpy as np

from src.nodes.receiver import TextPreprocessor, TextInput
from src.nodes.faiss_encoder import FaissTextEncoder  # Cambiado a FaissTextEncoder
from src.nodes.curator_graph import (
    CuratorAgent,
    FraudAnalysis,
    TextInput as CuratorTextInput,
)
from src.nodes.utils import FraudTypeRegistry


class WorkflowState(BaseModel):
    """State model for the fraud detection workflow."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    messages: List[BaseMessage]
    text_input: Optional[TextInput] = None
    similar_cases: Optional[List[Dict]] = None  # Cambiado para incluir scores
    analysis: Optional[FraudAnalysis] = None
    should_alert: bool = False
    new_type_name: Optional[str] = None

    def prepare_for_serialization(self):
        """Convert incompatible types to JSON-serializable types."""
        if self.similar_cases:
            for case in self.similar_cases:
                # Convert score to float if it exists
                if "score" in case and isinstance(case["score"], (np.float32, np.float64)):
                    case["score"] = float(case["score"])


class FraudDetectionConfig(BaseModel):
    """Configuration for the fraud detection workflow."""

    similarity_threshold: float = 0.85
    min_similar_cases: int = 3
    top_k_similar: int = 5  # Nuevo parámetro para FAISS


def create_workflow(
    preprocessor: TextPreprocessor,
    encoder: FaissTextEncoder,  # Cambiado a FaissTextEncoder
    curator: CuratorAgent,
    type_registry: FraudTypeRegistry,
    config: FraudDetectionConfig = FraudDetectionConfig(),
) -> StateGraph:
    """Create the fraud detection workflow graph."""

    def preprocess(state: dict) -> dict:
        """Preprocess incoming text."""
        try:
            state = WorkflowState.model_validate(state)
            messages = state.messages
            last_message = messages[-1]

            clean_text = last_message.content.replace("\n", " ").replace("\r", "")
            clean_text = clean_text.encode("ascii", "ignore").decode("ascii")

            text_input = preprocessor.process_text(
                clean_text,
                source="user_input",
                metadata={"timestamp": datetime.now().isoformat()},
            )

            return WorkflowState(
                messages=messages, text_input=text_input, should_alert=False
            ).model_dump()
        except Exception as e:
            print(f"Preprocessing error: {e}")
            raise

    def encode(state: dict) -> dict:
        """Find similar cases using FAISS."""
        try:
            state = WorkflowState.model_validate(state)

            if not state.text_input:
                return state.model_dump()

            # Buscar casos similares usando FAISS
            similar_cases = encoder.search_similar_cases(
                state.text_input.text, k=config.top_k_similar
            )

            # Filtrar casos por umbral de similitud
            similar_cases = [
                case
                for case in similar_cases
                if case["score"] >= config.similarity_threshold
            ]

            workflow_state = WorkflowState(
            messages=state.messages,
            text_input=state.text_input,
            similar_cases=similar_cases,
            should_alert=False,
        )
            workflow_state.prepare_for_serialization()
            return workflow_state.model_dump()
        except Exception as e:
            print(f"Encoding error: {e}")
            raise

    def curate(state: dict) -> dict:
        """Detailed fraud analysis with FAISS storage."""
        try:
            state = WorkflowState.model_validate(state)

            if not state.text_input:
                return state.model_dump()

            curator_input = CuratorTextInput(text=state.text_input.text)

            # Extraer solo los textos de los casos similares para el análisis
            similar_texts = [
                case["metadata"].get("text", "") for case in (state.similar_cases or [])
            ]

            print(f"Similar cases: {similar_texts}")

            analysis = curator.analyze_case(curator_input, similar_texts)

            # Almacenar en FAISS si es fraude
            if analysis.is_fraud:
                case_id = f"fraud_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                encoder.add_to_index(
                    state.text_input.text,
                    metadata={
                        "text": state.text_input.text,
                        "fraud_type": analysis.fraud_type,
                        "case_id": case_id,
                    },
                )

                # Guardar el índice FAISS
                encoder.save_faiss_index()

            return WorkflowState(
                messages=state.messages,
                text_input=state.text_input,
                similar_cases=state.similar_cases,
                analysis=analysis,
                should_alert=analysis.is_fraud,
                new_type_name=analysis.new_type_name,
            ).model_dump()
        except Exception as e:
            print(f"Curation error: {e}")
            raise

    # Create graph
    workflow = StateGraph(WorkflowState)

    workflow.add_node("preprocess", preprocess)
    workflow.add_node("encode", encode)
    workflow.add_node("curate", curate)

    workflow.add_edge("preprocess", "encode")
    workflow.add_edge("encode", "curate")
    workflow.add_edge("curate", END)

    workflow.set_entry_point("preprocess")

    return workflow.compile()


def run_fraud_detection(
    text: str, config: FraudDetectionConfig = FraudDetectionConfig()
) -> Dict:
    """Run the fraud detection workflow with configuration."""
    try:
        # Initialize components
        preprocessor = TextPreprocessor()
        encoder = FaissTextEncoder()
        curator = CuratorAgent()
        type_registry = FraudTypeRegistry()

        # Create workflow
        workflow = create_workflow(
            preprocessor, encoder, curator, type_registry, config
        )

        initial_state = WorkflowState(
            messages=[HumanMessage(content=text)], should_alert=False
        )

        final_state = workflow.invoke(initial_state.model_dump())
        final_state = WorkflowState.model_validate(final_state)

        new_type_name = None
        if final_state.new_type_name:
            type_registry.add_type(final_state.new_type_name)
            new_type_name = final_state.new_type_name

        return {
            "is_fraud": (
                final_state.analysis.is_fraud if final_state.analysis else False
            ),
            "fraud_type": (
                final_state.analysis.fraud_type if final_state.analysis else None
            ),
            "explanation": (
                final_state.analysis.explanation if final_state.analysis else ""
            ),
            "should_alert": final_state.should_alert,
            "new_type_name": new_type_name,
            "similar_cases_count": len(final_state.similar_cases or []),
        }

    except Exception as e:
        print(f"Error in fraud detection: {e}")
        return {
            "error": str(e),
            "is_fraud": False,
            "fraud_type": None,
            "explanation": "Error processing case",
            "should_alert": False,
            "similar_cases_count": 0,
        }
