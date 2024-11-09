Sistema de Detección y Clasificación de Fraudes con Machine Learning
Este proyecto implementa un sistema automatizado para la detección y clasificación de modalidades de fraude utilizando técnicas de procesamiento de lenguaje natural y machine learning. El sistema analiza testimonios de usuarios para identificar patrones de fraude y generar alertas en tiempo real.
🎯 Características Principales

Detección automática de fraudes basada en similitud de casos
Clasificación de nuevas modalidades de fraude
Análisis avanzado mediante LLMs para casos ambiguos
Sistema de actualización continua del corpus de fraudes
Generación automática de alertas

🛠️ Requisitos
bashCopypip install -r requirements.txt
Contenido de requirements.txt:
Copylangchain
langgraph
pydantic
numpy
ollama
jupyter
🚀 Guía de Implementación en Jupyter Notebook
pythonCopy# fraud_detection.ipynb

# 1. Importaciones necesarias
import sys
import os
from datetime import datetime
from typing import Dict, List

# Asegúrate de que el directorio del proyecto esté en el PYTHONPATH
project_root = "."  # Ajusta esto a la ruta de tu proyecto
sys.path.append(project_root)

from nodes.receiver import TextPreprocessor, TextInput
from nodes.encoder import TextEncoder
from nodes.curator import CuratorAgent
from nodes.utils import EmbeddingStorage, FraudTypeRegistry
from workflow import create_workflow, FraudDetectionConfig, run_fraud_detection

# 2. Configuración inicial
def setup_environment():
    """Configurar el entorno y directorios necesarios"""
    os.makedirs("data/embeddings", exist_ok=True)
    os.makedirs("data/fraud_types", exist_ok=True)
    return "Entorno configurado correctamente"

# 3. Inicialización de componentes
preprocessor = TextPreprocessor()
encoder = TextEncoder()
curator = CuratorAgent()
embedding_storage = EmbeddingStorage()
type_registry = FraudTypeRegistry()

# 4. Configuración del workflow
config = FraudDetectionConfig(
    similarity_threshold=0.85,
    min_similar_cases=3,
    confidence_threshold=0.7
)

# 5. Crear el workflow
workflow = create_workflow(
    preprocessor,
    encoder,
    curator,
    embedding_storage,
    type_registry,
    config
)

# 6. Función de prueba
def test_fraud_detection(text: str) -> Dict:
    """Función para probar la detección de fraude"""
    result = run_fraud_detection(text, config)
    return {
        "is_fraud": result["is_fraud"],
        "confidence": result["confidence"],
        "fraud_type": result["fraud_type"],
        "explanation": result["explanation"],
        "should_alert": result["should_alert"],
        "similar_cases_count": result["similar_cases_count"]
    }

# 7. Ejemplos de uso
ejemplo_texto = """
Me contactaron diciendo que habían detectado actividad sospechosa en mi cuenta bancaria 
y necesitaban verificar mi identidad. Me pidieron acceso remoto a mi computadora para 
"proteger mis fondos" y terminé perdiendo acceso a mis cuentas.
"""

resultado = test_fraud_detection(ejemplo_texto)
print("Resultado del análisis:", resultado)
📁 Estructura del Proyecto
Copyfraud_detection/
├── data/
│   ├── embeddings/     # Almacenamiento de embeddings
│   └── fraud_types/    # Registro de tipos de fraude
├── nodes/
│   ├── receiver.py     # Preprocesamiento de texto
│   ├── encoder.py      # Generación de embeddings
│   ├── curator.py      # Análisis de fraude
│   └── utils.py        # Utilidades y almacenamiento
├── workflow.py         # Definición del workflow
├── requirements.txt    # Dependencias
└── README.md          # Este archivo
🔍 Uso del Sistema

Configuración Inicial
pythonCopysetup_environment()

Análisis de un Caso
pythonCopytexto = "Descripción del posible fraude..."
resultado = test_fraud_detection(texto)

Interpretación de Resultados

is_fraud: Booleano indicando si se detectó fraude
confidence: Nivel de confianza de la detección
fraud_type: Tipo de fraude identificado
explanation: Explicación detallada
should_alert: Si se debe generar una alerta
similar_cases_count: Número de casos similares encontrados



⚙️ Configuración Avanzada
Puedes ajustar los parámetros del sistema modificando la configuración:
pythonCopyconfig = FraudDetectionConfig(
    similarity_threshold=0.85,  # Umbral de similitud para casos
    min_similar_cases=3,       # Mínimo de casos similares requeridos
    confidence_threshold=0.7    # Umbral de confianza para alertas
)
🤖 Integración con Ollama
El sistema utiliza Ollama para el análisis avanzado de casos. Asegúrate de tener Ollama instalado y configurado:
bashCopy# Instalar Ollama (si no está instalado)
curl https://ollama.ai/install.sh | sh

# Iniciar el servicio
ollama serve

# Descargar el modelo Mistral (usado por defecto)
ollama pull mistral
📊 Ejemplo de Output
pythonCopy{
    'is_fraud': True,
    'confidence': 0.92,
    'fraud_type': 'PHISHING',
    'explanation': 'Intento de phishing bancario con solicitud de acceso remoto...',
    'should_alert': True,
    'similar_cases_count': 5
}
🔒 Consideraciones de Seguridad

Los datos sensibles deben ser anonimizados antes de procesarlos
Implementar autenticación para el acceso al sistema
Mantener actualizadas todas las dependencias
Revisar regularmente los falsos positivos y negativos

📝 Notas Adicionales

El sistema aprende continuamente de nuevos casos
Los embeddings se almacenan localmente para análisis futuros
El registro de tipos de fraude se actualiza automáticamente
Las alertas deben ser revisadas antes de su difusión

🤝 Contribuciones
Las contribuciones son bienvenidas. Por favor, asegúrate de:

Hacer fork del repositorio
Crear una rama para tu feature
Commit y push de tus cambios
Crear un Pull Request

📄 Licencia
Este proyecto está bajo la licencia MIT.