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
    similarity_threshold: float = 0.85  # Umbral para considerar casos similares
    min_similar_cases: int = 3
    top_k_similar: int = 5  # Parámetro para FAISS
    auto_fraud_threshold: float = 0.2  # Si el promedio es menor que esto, es fraude automático


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

            # Calcular promedio de similaridad de los primeros 3 casos (o menos si no hay suficientes)
            if similar_cases:
                top_cases = similar_cases[:min(3, len(similar_cases))]
                avg_similarity = sum(case["score"] for case in top_cases) / len(top_cases)

                print(f"Average similarity: {avg_similarity}")

                # Si la similaridad promedio es muy baja (casos muy similares en FAISS)
                if avg_similarity <= config.auto_fraud_threshold and similar_cases:
                    print("Auto-fraud detected")
                    # Siempre usar el caso más similar (menor score) para el fraud_type
                    most_similar_case = min(similar_cases, key=lambda x: x["score"])

                    print(f"Most similar case: {most_similar_case}")

                    workflow_state = WorkflowState(
                        messages=state.messages,
                        text_input=state.text_input,
                        similar_cases=similar_cases,
                        analysis=FraudAnalysis(
                            is_fraud=True,
                            fraud_type=most_similar_case["metadata"]["fraud_type"],
                            explanation=f"Automatic classification based on similarity with case showing fraud type: {most_similar_case['metadata']['fraud_type']}",
                            similar_cases=[case["metadata"]["text"] for case in similar_cases],
                            timestamp=datetime.now().isoformat(),
                            new_type_name=None
                        ),
                        should_alert=True
                    )
                    workflow_state.prepare_for_serialization()
                    return workflow_state.model_dump()

            # Filtrar casos por umbral de similaridad para el curator
            similar_cases = [
                case
                for case in similar_cases
                if case["score"] <= config.similarity_threshold
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
                case["metadata"].get("fraud_type", "") for case in (state.similar_cases or [])
            ]

            similar_texts = [text for text in similar_texts if type(text) == str]

            print(f"Similar texts: {similar_texts}")

            analysis = curator.analyze_case(curator_input, similar_texts)

            # Almacenar en FAISS si es fraude
            if analysis.is_fraud:
                case_id = f"fraud_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                encoder.add_to_index(
                    state.text_input.text,
                    metadata={
                        "text": state.text_input.text,
                        "fraud_type": analysis.fraud_type if analysis.fraud_type != "NEW" else state.new_type_name,
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

    def should_use_curator(state: dict) -> str:
        """
        Determina si el caso debe ser procesado por el curator o finalizar el flujo.
        Returns:
            str: "curate" si debe usar el curator, END si debe finalizar el flujo
        """
        try:
            state = WorkflowState.model_validate(state)

            # Si no hay análisis previo (no fue clasificado automáticamente), usar curator
            if not state.analysis:
                return "curate"

            # Si ya fue clasificado como fraude automáticamente, terminar el flujo
            if state.analysis and state.analysis.is_fraud:
                return END

            return "curate"
        except Exception as e:
            print(f"Should use curator error: {e}")
            raise

    # Create graph
    workflow = StateGraph(WorkflowState)

    workflow.add_node("preprocess", preprocess)
    workflow.add_node("encode", encode)
    workflow.add_node("curate", curate)

    workflow.add_edge("preprocess", "encode")
    workflow.add_conditional_edges(
        "encode",
        should_use_curator,
        {
            "curate": "curate",
            END: END
        }
    )
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
