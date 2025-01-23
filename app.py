import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import PyPDF2
import docx
import os
import json
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class TextSegment:
    text: str
    source: str
    reference: Dict[str, str]

class DocumentAnalyzer:
    def __init__(self):
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
        self.segments: List[TextSegment] = []
        self.index = None
        self.embeddings = None
        
    def extract_pdf_info(self, pdf_path: str) -> List[TextSegment]:
        segments = []
        with open(pdf_path, 'rb') as file:
            pdf = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                # Dividir el texto en párrafos
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                for para_num, para in enumerate(paragraphs, 1):
                    if len(para.split()) >= 10:  # Solo párrafos con al menos 10 palabras
                        segments.append(TextSegment(
                            text=para,
                            source=os.path.basename(pdf_path),
                            reference={"página": str(page_num), "párrafo": str(para_num)}
                        ))
        return segments

    def extract_docx_info(self, docx_path: str) -> List[TextSegment]:
        segments = []
        doc = docx.Document(docx_path)
        current_heading = "Inicio del documento"
        
        for para_num, para in enumerate(doc.paragraphs, 1):
            if para.style.name.startswith('Heading'):
                current_heading = para.text
            elif len(para.text.split()) >= 10:  # Solo párrafos con al menos 10 palabras
                segments.append(TextSegment(
                    text=para.text,
                    source=os.path.basename(docx_path),
                    reference={"sección": current_heading, "párrafo": str(para_num)}
                ))
        return segments

    def process_document(self, file_path: str, file_type: str):
        if file_type == 'pdf':
            new_segments = self.extract_pdf_info(file_path)
        elif file_type == 'docx':
            new_segments = self.extract_docx_info(file_path)
        else:
            return
        
        self.segments.extend(new_segments)
        
        # Actualizar embeddings e índice
        texts = [seg.text for seg in self.segments]
        self.embeddings = self.model.encode(texts)
        
        # Crear o actualizar índice FAISS
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        else:
            self.index.reset()
        self.index.add(self.embeddings)
        
        # Guardar estado
        self.save_state()

    def get_recommendations(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.segments:
            return []
        
        query_vector = self.model.encode([query])
        distances, indices = self.index.search(query_vector, top_k)
        
        recommendations = []
        for idx in indices[0]:
            if idx < len(self.segments):
                segment = self.segments[idx]
                recommendations.append({
                    'texto': segment.text,
                    'fuente': segment.source,
                    'referencia': segment.reference
                })
        
        return recommendations

    def save_state(self):
        """Guardar el estado actual en archivos"""
        state_dir = 'app_state'
        os.makedirs(state_dir, exist_ok=True)
        
        # Guardar información de segmentos
        segments_data = [
            {'text': s.text, 'source': s.source, 'reference': s.reference}
            for s in self.segments
        ]
        with open(os.path.join(state_dir, 'segments.json'), 'w', encoding='utf-8') as f:
            json.dump(segments_data, f, ensure_ascii=False, indent=2)
        
        # Guardar embeddings
        if self.embeddings is not None:
            np.save(os.path.join(state_dir, 'embeddings.npy'), self.embeddings)

    def load_state(self):
        """Cargar el estado guardado"""
        state_dir = 'app_state'
        if not os.path.exists(state_dir):
            return
        
        # Cargar segmentos
        segments_path = os.path.join(state_dir, 'segments.json')
        if os.path.exists(segments_path):
            with open(segments_path, 'r', encoding='utf-8') as f:
                segments_data = json.load(f)
                self.segments = [
                    TextSegment(
                        text=s['text'],
                        source=s['source'],
                        reference=s['reference']
                    )
                    for s in segments_data
                ]
        
        # Cargar embeddings
        embeddings_path = os.path.join(state_dir, 'embeddings.npy')
        if os.path.exists(embeddings_path):
            self.embeddings = np.load(embeddings_path)
            if self.embeddings is not None:
                self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
                self.index.add(self.embeddings)

def main():
    st.set_page_config(
        page_title="Análisis de Documentos",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Sistema de Análisis y Recomendación de Documentos")
    
    # Inicializar o cargar el analizador
    analyzer = DocumentAnalyzer()
    analyzer.load_state()
    
    # Sidebar para carga de documentos
    with st.sidebar:
        st.header("Cargar Documentos")
        uploaded_files = st.file_uploader(
            "Selecciona documentos PDF o DOCX",
            type=['pdf', 'docx'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                with st.spinner(f"Procesando {uploaded_file.name}..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        file_path = tmp_file.name
                    
                    analyzer.process_document(
                        file_path,
                        uploaded_file.name.split('.')[-1]
                    )
                    os.unlink(file_path)
                st.success(f"✅ {uploaded_file.name} procesado")
        
        st.markdown("---")
        st.markdown("### Documentos Procesados")
        processed_docs = set(seg.source for seg in analyzer.segments)
        for doc in processed_docs:
            st.markdown(f"- {doc}")
    
    # Área principal
    st.header("🔍 Buscar en los Documentos")
    
    # Configuración de búsqueda
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Escribe tu consulta:", placeholder="¿Qué quieres encontrar?")
    with col2:
        num_results = st.selectbox("Número de resultados:", [4, 5, 6], index=1)
    
    # Mostrar resultados
    if query:
        recommendations = analyzer.get_recommendations(query, num_results)
        
        if recommendations:
            st.markdown("### 📑 Resultados Encontrados")
            
            for i, rec in enumerate(recommendations, 1):
                with st.expander(f"Resultado {i} - {rec['fuente']}", expanded=True):
                    st.markdown(f"**Texto encontrado:**")
                    st.markdown(rec['texto'])
                    st.markdown("**Referencia:**")
                    for key, value in rec['referencia'].items():
                        st.markdown(f"- {key.title()}: {value}")
        else:
            st.warning("No se encontraron resultados. Intenta reformular tu consulta.")

if __name__ == "__main__":
    main()