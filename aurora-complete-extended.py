#!/usr/bin/env python3
"""
Aurora Complete System - Vollständig integriertes System mit Ollama und erweitertem Langzeitspeicher
Alle Plugins funktionsfähig, persistente SQL-Speicherung, semantische Suche, Selbstentwicklung
VOLLSTÄNDIGE OLLAMA INTEGRATION MIT LOKALEN MODELLEN UND LANGZEITSPEICHER
"""

import sys
import os
import json
import time
import asyncio
import logging
import requests
import threading
import uuid
import traceback
import numpy as np
import pickle
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
from enum import Enum

# --- BUGFIX: Patch for huggingface_hub's cached_download removal ---
# This resolves an ImportError in older sentence-transformers versions
# that rely on a function renamed in newer huggingface_hub versions.
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, 'cached_download'):
        huggingface_hub.cached_download = huggingface_hub.hf_hub_download
except (ImportError, AttributeError):
    pass # If libraries are not installed, let it fail later

# Database and ML imports
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Text, Boolean, JSON, ForeignKey
from sqlalchemy.orm import sessionmaker, Session, relationship, scoped_session, DeclarativeBase
from sqlalchemy.pool import QueuePool
import hashlib
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QSlider, QGroupBox, QGridLayout,
    QFrame, QScrollArea, QSplitter, QTabWidget, QProgressBar,
    QLineEdit, QComboBox, QListWidget, QTextBrowser, QSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QTreeWidget,
    QTreeWidgetItem, QCheckBox, QRadioButton, QButtonGroup, QMessageBox,
    QInputDialog
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation,
    QEasingCurve, QRect, QPoint, pyqtProperty, QSize,
    QObject, QMutex, QMutexLocker
)
from PyQt6.QtGui import (
    QPalette, QColor, QFont, QLinearGradient, QPainter,
    QBrush, QPen, QFontDatabase, QPixmap, QPainterPath,
    QRadialGradient
)

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =====================================
# CONSTANTS & CONFIGURATION
# =====================================

# Aurora Memory Storage Path
# ================================================================================
# DYNAMISCHE PFADKONFIGURATION (Ersetzt die feste C:\ Adresse)
# ================================================================================
BASE_DIR = Path(__file__).resolve().parent
_storage_dir = BASE_DIR / "aurora_data"
_storage_dir.mkdir(parents=True, exist_ok=True)
AURORA_STORAGE_PATH = str(_storage_dir)
# ================================================================================
# Database path
DATABASE_PATH = os.path.join(AURORA_STORAGE_PATH, "aurora_memory.db")
EMBEDDINGS_PATH = os.path.join(AURORA_STORAGE_PATH, "embeddings")
CHECKPOINTS_PATH = os.path.join(AURORA_STORAGE_PATH, "checkpoints")
KNOWLEDGE_PATH = os.path.join(AURORA_STORAGE_PATH, "knowledge_base")

# Create subdirectories
for path in [EMBEDDINGS_PATH, CHECKPOINTS_PATH, KNOWLEDGE_PATH]:
    os.makedirs(path, exist_ok=True)

# Set custom model path if needed
OLLAMA_MODEL_PATH = r"C:\Users\marce\Desktop\Aurora Lora training complete\Aurora Metacognition Environment\models"
if os.path.exists(OLLAMA_MODEL_PATH):
    os.environ['OLLAMA_MODELS'] = OLLAMA_MODEL_PATH
    logger.info(f"Set OLLAMA_MODELS to: {OLLAMA_MODEL_PATH}")

COLORS = {
    'bg_primary': '#0d1117',
    'bg_secondary': '#161b22',
    'bg_tertiary': '#21262d',
    'border': '#30363d',
    'text_primary': '#c9d1d9',
    'text_secondary': '#8b949e',
    'accent_blue': '#58a6ff',
    'accent_green': '#3fb950',
    'accent_red': '#f85149',
    'accent_yellow': '#d29922',
    'accent_purple': '#a371f7',
    'accent_cyan': '#79c0ff',
    'accent_orange': '#ff7b00',
    'gradient_1': '#1f6feb',
    'gradient_2': '#388bfd',
}

DIMENSION_COLORS = {
    'D0': '#ff6b6b',  # Cognitive
    'D1': '#4ecdc4',  # Emotional
    'D2': '#45b7d1',  # Creative
    'D3': '#96ceb4',  # Memory
    'D4': '#ffeaa7',  # Intuition
    'D5': '#dfe6e9',  # Meta
    'D6': '#a29bfe',  # Unified
}

# =====================================
# DATABASE MODELS
# =====================================

class Base(DeclarativeBase):
    pass

class MemoryEntry(Base):
    """Persistente Speicher-Einträge in SQL"""
    __tablename__ = 'memories'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    content = Column(Text, nullable=False)
    memory_type = Column(String(50))  # episodic, semantic, procedural, emotional, meta, associative, contextual
    category = Column(String(100))  # Unterkategorie
    timestamp = Column(DateTime, default=datetime.now)
    importance = Column(Float, default=0.5)
    emotional_valence = Column(Float, default=0.5)
    access_count = Column(Integer, default=0)
    consolidation_count = Column(Integer, default=0)
    decay_rate = Column(Float, default=0.95)
    embedding_file = Column(String)  # Pfad zur Embedding-Datei
    # --- BUGFIX: Renamed 'metadata' to 'meta_data' to avoid SQLAlchemy reserved keyword conflict ---
    meta_data = Column(JSON)
    
    # Relationships
    associations = relationship("MemoryAssociation", foreign_keys="MemoryAssociation.source_id", back_populates="source")
    patterns = relationship("PatternMemory", back_populates="memory")
    reflections = relationship("MetaReflection", back_populates="memory")

class MemoryAssociation(Base):
    """Assoziationen zwischen Memories"""
    __tablename__ = 'associations'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    source_id = Column(String, ForeignKey('memories.id'))
    target_id = Column(String, ForeignKey('memories.id'))
    association_type = Column(String(50))  # semantic, temporal, causal, emotional
    strength = Column(Float, default=0.5)
    created_at = Column(DateTime, default=datetime.now)
    
    source = relationship("MemoryEntry", foreign_keys=[source_id], back_populates="associations")
    target = relationship("MemoryEntry", foreign_keys=[target_id])

class PatternMemory(Base):
    """Erkannte Muster und Patterns"""
    __tablename__ = 'patterns'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    memory_id = Column(String, ForeignKey('memories.id'))
    pattern_type = Column(String(50))  # behavioral, linguistic, cognitive, emotional
    pattern_content = Column(Text)
    frequency = Column(Integer, default=1)
    confidence = Column(Float, default=0.5)
    discovered_at = Column(DateTime, default=datetime.now)
    
    memory = relationship("MemoryEntry", back_populates="patterns")

class MetaReflection(Base):
    """Meta-kognitive Reflexionen und Selbstentwicklung"""
    __tablename__ = 'reflections'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    memory_id = Column(String, ForeignKey('memories.id'), nullable=True)
    reflection_type = Column(String(50))  # learning, improvement, insight, strategy
    content = Column(Text)
    insights = Column(JSON)
    improvement_suggestions = Column(JSON)
    created_at = Column(DateTime, default=datetime.now)
    applied = Column(Boolean, default=False)
    effectiveness = Column(Float, nullable=True)
    
    memory = relationship("MemoryEntry", back_populates="reflections")

class KnowledgeNode(Base):
    """Wissensgraph-Knoten"""
    __tablename__ = 'knowledge_nodes'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    concept = Column(String(200), unique=True)
    definition = Column(Text)
    category = Column(String(100))
    confidence = Column(Float, default=0.5)
    learned_from = Column(JSON)  # Liste von Memory-IDs
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    edges = relationship("KnowledgeEdge", foreign_keys="KnowledgeEdge.source_id", back_populates="source")

class KnowledgeEdge(Base):
    """Verbindungen im Wissensgraph"""
    __tablename__ = 'knowledge_edges'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    source_id = Column(String, ForeignKey('knowledge_nodes.id'))
    target_id = Column(String, ForeignKey('knowledge_nodes.id'))
    relationship_type = Column(String(50))  # is_a, part_of, causes, related_to
    strength = Column(Float, default=0.5)
    
    source = relationship("KnowledgeNode", foreign_keys=[source_id], back_populates="edges")
    target = relationship("KnowledgeNode", foreign_keys=[target_id])

class ConversationSession(Base):
    """Konversations-Sessions für Kontext"""
    __tablename__ = 'sessions'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    started_at = Column(DateTime, default=datetime.now)
    ended_at = Column(DateTime, nullable=True)
    total_messages = Column(Integer, default=0)
    emotional_trajectory = Column(JSON)
    topics_discussed = Column(JSON)
    key_insights = Column(JSON)

# =====================================
# OLLAMA INTEGRATION
# =====================================

class OllamaProvider:
    """Vollständige Ollama Integration mit lokalen Modellen"""
    
    def __init__(self):
        self.endpoint = "http://localhost:11434"
        self.available = False
        self.models = []
        self.selected_model = None
        self.check_connection()
        
    def check_connection(self) -> bool:
        """Prüft Ollama Verbindung"""
        try:
            response = requests.get(f"{self.endpoint}/api/version", timeout=3)
            if response.status_code == 200:
                self.available = True
                version_info = response.json()
                logger.info(f"✅ Ollama online: {version_info}")
                self.list_models()
                return True
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️ Ollama nicht verfügbar: {e}")
        except Exception as e:
            logger.error(f"Unerwarteter Fehler bei Ollama-Verbindung: {e}")
            
        self.available = False
        return False
        
    def list_models(self) -> List[str]:
        """Listet verfügbare Modelle"""
        if not self.available:
            return []
            
        try:
            response = requests.get(f"{self.endpoint}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models_data = data.get("models", [])
                
                self.models = []
                for model in models_data:
                    model_name = model.get("name", "")
                    if model_name:
                        self.models.append(model_name)
                        logger.info(f"  📦 Model found: {model_name}")
                
                if self.models and not self.selected_model:
                    self.selected_model = self.models[0]
                    logger.info(f"  ➡️ Selected default model: {self.selected_model}")
                    
                return self.models
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Modelle: {e}")
            
        return []
        
    def generate(self, prompt: str, model: str = None, 
                temperature: float = 0.7, max_tokens: int = 2000, 
                stream: bool = False) -> str:
        """Generiert Antwort mit Ollama"""
        if not self.available:
            return "⚠️ Ollama ist nicht verfügbar. Bitte starte 'ollama serve' im Terminal."
            
        use_model = model or self.selected_model or (self.models[0] if self.models else "mistral:latest")
        
        try:
            logger.info(f"🚀 Generating with model: {use_model}")
            logger.debug(f"Prompt (first 200 chars): {prompt[:200]}...")
            
            data = {
                "model": use_model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "num_ctx": 4096,
                    "top_k": 40,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            }
            
            response = requests.post(
                f"{self.endpoint}/api/generate",
                json=data,
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "")
                
                if generated_text:
                    logger.info(f"✅ Generation successful: {len(generated_text)} chars")
                    
                    if "total_duration" in result:
                        total_ms = result["total_duration"] / 1_000_000
                        logger.debug(f"Generation took: {total_ms:.2f}ms")
                        
                    return generated_text
                else:
                    logger.warning("Leere Antwort von Ollama erhalten")
                    return "Ollama hat eine leere Antwort generiert. Bitte versuche es erneut."
            else:
                error_msg = f"Ollama API Fehler: Status {response.status_code}"
                logger.error(f"{error_msg}\nResponse: {response.text}")
                return f"⚠️ {error_msg}"
                
        except requests.exceptions.Timeout:
            logger.error("Ollama Generation Timeout")
            return "⏱️ Zeitüberschreitung bei der Generierung. Versuche eine kürzere Anfrage."
        except requests.exceptions.RequestException as e:
            logger.error(f"Request Fehler bei Ollama-Generierung: {e}")
            return f"⚠️ Verbindungsfehler: {str(e)}"
        except Exception as e:
            logger.error(f"Unerwarteter Fehler bei Ollama-Generierung: {e}\n{traceback.format_exc()}")
            return f"⚠️ Fehler: {str(e)}"
            
    def set_model(self, model_name: str):
        """Setzt das aktive Modell"""
        if model_name in self.models:
            self.selected_model = model_name
            logger.info(f"Model gewechselt zu: {model_name}")
            return True
        return False
        
    def pull_model(self, model_name: str) -> bool:
        """Lädt ein neues Modell herunter"""
        try:
            logger.info(f"📥 Pulling model: {model_name}")
            response = requests.post(
                f"{self.endpoint}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=600
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            status = data.get("status", "")
                            logger.info(f"  {status}")
                        except json.JSONDecodeError:
                            logger.warning(f"Could not decode JSON from line: {line}")
                        
                logger.info(f"✅ Model {model_name} successfully pulled")
                self.list_models()
                return True
                
        except Exception as e:
            logger.error(f"Fehler beim Model-Download: {e}")
            
        return False

# =====================================
# ENHANCED MEMORY SYSTEM
# =====================================

class EnhancedMemorySystem:
    """Aurora LZ-Memory System mit SQL-Persistenz und semantischer Suche"""
    
    def __init__(self):
        # Database setup
        self.engine = create_engine(
            f'sqlite:///{DATABASE_PATH}',
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            echo=False
        )
        Base.metadata.create_all(self.engine)
        
        Session = scoped_session(sessionmaker(bind=self.engine))
        self.session = Session()
        
        # Embedding model (using sentence-transformers)
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Embedding model loaded")
        except Exception as e:
            logger.error(f"Could not load embedding model: {e}")
            self.embedding_model = None
        
        # Memory allocations
        self.stm_allocation = 15  # MB
        self.ltm_allocation = 54  # MB
        self.cache_hit_rate = 0.92
        
        # Memory stores (in-memory caches)
        self.working_memory = deque(maxlen=10)
        self.stm_cache = deque(maxlen=100)
        
        # Pattern detection
        self.pattern_detector = PatternDetector()
        
        # Self-development tracker
        self.development_tracker = SelfDevelopmentTracker(self.session)
        
        # Current session
        self.current_session = self._create_session()
        
        logger.info(f"🧠 Enhanced Aurora Memory System initialized at: {AURORA_STORAGE_PATH}")
        
    def _create_session(self) -> ConversationSession:
        """Erstellt neue Konversations-Session"""
        session = ConversationSession()
        self.session.add(session)
        self.session.commit()
        return session
        
    def store_memory(self, content: str, memory_type: str = "episodic", 
                    category: str = None, importance: float = 0.5,
                    metadata: Dict = None) -> str:
        """Speichert neue Erinnerung persistent in SQL"""
        
        # Generate embeddings
        embedding_file = None
        embeddings = None
        if self.embedding_model:
            embeddings = self.embedding_model.encode([content])[0]
            embedding_file = self._save_embeddings(embeddings)
        
        # Create memory entry
        memory = MemoryEntry(
            id=str(uuid.uuid4()),
            content=content,
            memory_type=memory_type,
            category=category or "general",
            importance=importance,
            embedding_file=embedding_file,
            meta_data=metadata or {}
        )
        
        # Add to database
        self.session.add(memory)
        self.session.commit()
        
        # Add to working memory cache
        self.working_memory.append({
            "id": memory.id,
            "content": content,
            "embeddings": embeddings if self.embedding_model else None
        })
        
        # Detect patterns
        self._detect_and_store_patterns(memory)
        
        # Create associations
        self._create_associations(memory)
        
        # Update session
        self.current_session.total_messages += 1
        self.session.commit()
        
        logger.info(f"💾 Memory stored: {memory.id} (type: {memory_type}, importance: {importance:.2f})")
        return memory.id
        
    def _save_embeddings(self, embeddings: np.ndarray) -> str:
        """Speichert Embeddings als Datei"""
        filename = f"{uuid.uuid4()}.npy"
        filepath = os.path.join(EMBEDDINGS_PATH, filename)
        np.save(filepath, embeddings)
        return filename
        
    def _load_embeddings(self, filename: str) -> Optional[np.ndarray]:
        """Lädt Embeddings aus Datei"""
        if not filename:
            return None
        filepath = os.path.join(EMBEDDINGS_PATH, filename)
        if os.path.exists(filepath):
            return np.load(filepath)
        return None
        
    def _detect_and_store_patterns(self, memory: MemoryEntry):
        """Erkennt und speichert Muster"""
        patterns = self.pattern_detector.detect(memory.content)
        
        for pattern_type, pattern_content, confidence in patterns:
            # Check if pattern exists
            existing = self.session.query(PatternMemory).filter_by(
                pattern_content=pattern_content
            ).first()
            
            if existing:
                existing.frequency += 1
                existing.confidence = max(existing.confidence, confidence)
            else:
                pattern = PatternMemory(
                    id=str(uuid.uuid4()),
                    memory_id=memory.id,
                    pattern_type=pattern_type,
                    pattern_content=pattern_content,
                    confidence=confidence
                )
                self.session.add(pattern)
                
        self.session.commit()
        
    def _create_associations(self, memory: MemoryEntry):
        """Erstellt Assoziationen zu ähnlichen Memories"""
        if not self.embedding_model:
            return
            
        # Get recent memories for association
        recent_memories = self.session.query(MemoryEntry).order_by(
            MemoryEntry.timestamp.desc()
        ).limit(20).all()
        
        memory_embeddings = self._load_embeddings(memory.embedding_file)
        if memory_embeddings is None:
            return
            
        for other_memory in recent_memories:
            if other_memory.id == memory.id:
                continue
                
            other_embeddings = self._load_embeddings(other_memory.embedding_file)
            if other_embeddings is None:
                continue
                
            # Calculate similarity
            similarity = cosine_similarity(
                [memory_embeddings], [other_embeddings]
            )[0][0]
            
            if similarity > 0.7:  # High similarity threshold
                association = MemoryAssociation(
                    id=str(uuid.uuid4()),
                    source_id=memory.id,
                    target_id=other_memory.id,
                    association_type="semantic",
                    strength=float(similarity)
                )
                self.session.add(association)
                
        self.session.commit()
        
    def semantic_search(self, query: str, limit: int = 5) -> List[Dict]:
        """Semantische Suche mit Embeddings über SQL"""
        if not self.embedding_model:
            # Fallback to keyword search
            return self._keyword_search(query, limit)
            
        # Generate query embeddings
        query_embeddings = self.embedding_model.encode([query])[0]
        
        # Get all memories with embeddings
        memories = self.session.query(MemoryEntry).filter(
            MemoryEntry.embedding_file.isnot(None)
        ).all()
        
        results = []
        for memory in memories:
            memory_embeddings = self._load_embeddings(memory.embedding_file)
            if memory_embeddings is None:
                continue
                
            similarity = cosine_similarity(
                [query_embeddings], [memory_embeddings]
            )[0][0]
            
            relevance = similarity * memory.importance * (1 + memory.access_count * 0.1)
            
            results.append({
                "memory": memory,
                "similarity": float(similarity),
                "relevance": float(relevance)
            })
            
        # Sort by relevance
        results.sort(key=lambda x: x["relevance"], reverse=True)
        
        # Update access counts
        for result in results[:limit]:
            result["memory"].access_count += 1
            
        self.session.commit()
        
        return results[:limit]
        
    def _keyword_search(self, query: str, limit: int) -> List[Dict]:
        """Fallback Keyword-Suche"""
        keywords = query.lower().split()
        
        memories = self.session.query(MemoryEntry).all()
        results = []
        
        for memory in memories:
            content_lower = memory.content.lower()
            matches = sum(1 for keyword in keywords if keyword in content_lower)
            
            if matches > 0:
                results.append({
                    "memory": memory,
                    "similarity": matches / len(keywords),
                    "relevance": (matches / len(keywords)) * memory.importance
                })
                
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:limit]
        
    def consolidate_memories(self):
        """Konsolidiert Memories und entwickelt sich selbst weiter"""
        logger.info("🔄 Starting memory consolidation and self-development...")
        
        # Get memories for consolidation
        stm_memories = self.session.query(MemoryEntry).filter(
            MemoryEntry.consolidation_count < 3,
            MemoryEntry.importance > 0.3
        ).order_by(MemoryEntry.timestamp.desc()).limit(50).all()
        
        consolidated = 0
        patterns_found = 0
        
        for memory in stm_memories:
            memory.consolidation_count += 1
            
            # Increase importance based on access patterns
            if memory.access_count > 3:
                memory.importance = min(1.0, memory.importance * 1.2)
                consolidated += 1
                
            # Apply decay to unused memories
            if memory.access_count == 0:
                memory.importance *= memory.decay_rate
                
        # Extract new patterns across memories
        all_patterns = self.session.query(PatternMemory).all()
        pattern_groups = {}
        
        for pattern in all_patterns:
            if pattern.pattern_content not in pattern_groups:
                pattern_groups[pattern.pattern_content] = []
            pattern_groups[pattern.pattern_content].append(pattern)
            
        # Create meta-reflections for frequent patterns
        for pattern_content, patterns in pattern_groups.items():
            if len(patterns) >= 3:  # Pattern appears at least 3 times
                self._create_meta_reflection(pattern_content, patterns)
                patterns_found += 1
                
        # Self-development analysis
        insights = self.development_tracker.analyze_performance()
        if insights:
            self._apply_development_insights(insights)
            
        self.session.commit()
        
        logger.info(f"✅ Consolidation complete: {consolidated} memories consolidated, {patterns_found} patterns found")
        
    def _create_meta_reflection(self, pattern_content: str, patterns: List[PatternMemory]):
        """Erstellt Meta-Reflexion basierend auf Mustern"""
        reflection = MetaReflection(
            id=str(uuid.uuid4()),
            reflection_type="pattern_insight",
            content=f"Recurring pattern detected: {pattern_content}",
            insights={
                "pattern": pattern_content,
                "frequency": len(patterns),
                "confidence": np.mean([p.confidence for p in patterns])
            },
            improvement_suggestions={
                "action": "optimize_response",
                "focus_area": pattern_content
            }
        )
        self.session.add(reflection)
        
    def _apply_development_insights(self, insights: Dict):
        """Wendet Entwicklungs-Insights an"""
        for insight_type, data in insights.items():
            reflection = MetaReflection(
                id=str(uuid.uuid4()),
                reflection_type="self_improvement",
                content=f"Self-development insight: {insight_type}",
                insights=data,
                improvement_suggestions=data.get("suggestions", {})
            )
            self.session.add(reflection)
            
    def update_knowledge_graph(self, concept: str, definition: str, 
                              category: str, related_memories: List[str]):
        """Aktualisiert den Wissensgraph"""
        # Check if concept exists
        node = self.session.query(KnowledgeNode).filter_by(concept=concept).first()
        
        if node:
            node.definition = definition
            node.confidence = min(1.0, node.confidence * 1.1)
            node.updated_at = datetime.now()
        else:
            node = KnowledgeNode(
                id=str(uuid.uuid4()),
                concept=concept,
                definition=definition,
                category=category,
                learned_from=related_memories
            )
            self.session.add(node)
            
        self.session.commit()
        
    def get_statistics(self) -> Dict:
        """Gibt detaillierte Memory-Statistiken zurück"""
        total_memories = self.session.query(MemoryEntry).count()
        memory_types = {}
        
        for mem_type in ["episodic", "semantic", "procedural", "emotional", "meta"]:
            count = self.session.query(MemoryEntry).filter_by(memory_type=mem_type).count()
            memory_types[mem_type] = count
            
        patterns_count = self.session.query(PatternMemory).count()
        associations_count = self.session.query(MemoryAssociation).count()
        reflections_count = self.session.query(MetaReflection).count()
        knowledge_nodes = self.session.query(KnowledgeNode).count()
        
        return {
            "working_memory": len(self.working_memory),
            "stm_count": len(self.stm_cache),
            "ltm_count": total_memories,
            "archive_count": 0,  # Not implemented yet
            "total_memories": total_memories,
            "memory_types": memory_types,
            "patterns_count": patterns_count,
            "associations_count": associations_count,
            "reflections_count": reflections_count,
            "knowledge_nodes": knowledge_nodes,
            "session_messages": self.current_session.total_messages if self.current_session else 0,
            "stm_allocation_mb": self.stm_allocation,
            "ltm_allocation_mb": self.ltm_allocation,
            "cache_hit_rate": self.cache_hit_rate,
            "storage_path": AURORA_STORAGE_PATH,
            "database_size_mb": os.path.getsize(DATABASE_PATH) / (1024 * 1024) if os.path.exists(DATABASE_PATH) else 0
        }
        
    def export_knowledge(self, filepath: str = None):
        """Exportiert Wissen als JSON"""
        if not filepath:
            filepath = os.path.join(KNOWLEDGE_PATH, f"aurora_knowledge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
        knowledge = {
            "memories": [],
            "patterns": [],
            "reflections": [],
            "knowledge_graph": []
        }
        
        # Export memories
        memories = self.session.query(MemoryEntry).all()
        for memory in memories:
            knowledge["memories"].append({
                "id": memory.id,
                "content": memory.content,
                "type": memory.memory_type,
                "importance": memory.importance,
                "timestamp": memory.timestamp.isoformat()
            })
            
        # Export patterns
        patterns = self.session.query(PatternMemory).all()
        for pattern in patterns:
            knowledge["patterns"].append({
                "type": pattern.pattern_type,
                "content": pattern.pattern_content,
                "frequency": pattern.frequency,
                "confidence": pattern.confidence
            })
            
        # Export reflections
        reflections = self.session.query(MetaReflection).all()
        for reflection in reflections:
            knowledge["reflections"].append({
                "type": reflection.reflection_type,
                "content": reflection.content,
                "insights": reflection.insights
            })
            
        # Export knowledge graph
        nodes = self.session.query(KnowledgeNode).all()
        for node in nodes:
            knowledge["knowledge_graph"].append({
                "concept": node.concept,
                "definition": node.definition,
                "category": node.category,
                "confidence": node.confidence
            })
            
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(knowledge, f, indent=2, ensure_ascii=False)
            
        logger.info(f"📁 Knowledge exported to: {filepath}")
        return filepath

# =====================================
# PATTERN DETECTION
# =====================================

class PatternDetector:
    """Detects patterns in text and behavior"""
    
    def detect(self, text: str) -> List[Tuple[str, str, float]]:
        """Returns list of (pattern_type, pattern_content, confidence)"""
        patterns = []
        
        # Linguistic patterns
        words = text.lower().split()
        
        # Bi-grams
        if len(words) > 1:
            for i in range(len(words) - 1):
                bigram = f"{words[i]}_{words[i+1]}"
                patterns.append(("linguistic", bigram, 0.5))
            
        # Tri-grams
        if len(words) > 2:
            for i in range(len(words) - 2):
                trigram = f"{words[i]}_{words[i+1]}_{words[i+2]}"
                patterns.append(("linguistic", trigram, 0.3))
            
        # Emotional patterns
        emotional_words = {
            "positive": ["gut", "freude", "glücklich", "super", "toll", "schön", "liebe"],
            "negative": ["schlecht", "traurig", "wütend", "ärger", "problem", "schwierig"]
        }
        
        for emotion, emotion_words_list in emotional_words.items():
            count = sum(1 for word in emotion_words_list if word in text.lower())
            if count > 0:
                patterns.append(("emotional", emotion, min(1.0, count * 0.3)))
                
        # Cognitive patterns
        if "?" in text:
            patterns.append(("cognitive", "question", 0.8))
        if any(word in text.lower() for word in ["weil", "darum", "deshalb"]):
            patterns.append(("cognitive", "reasoning", 0.7))
            
        return patterns

# =====================================
# SELF-DEVELOPMENT TRACKER
# =====================================

class SelfDevelopmentTracker:
    """Tracks and manages Aurora's self-development"""
    
    def __init__(self, session: Session):
        self.session = session
        self.performance_metrics = {
            "response_quality": [],
            "learning_rate": [],
            "pattern_recognition": [],
            "emotional_understanding": []
        }
        
    def analyze_performance(self) -> Dict:
        """Analyzes performance and suggests improvements"""
        insights = {}
        
        # Analyze recent reflections
        recent_reflections = self.session.query(MetaReflection).filter(
            MetaReflection.applied == False
        ).order_by(MetaReflection.created_at.desc()).limit(10).all()
        
        if recent_reflections:
            # Group by type
            reflection_types = {}
            for reflection in recent_reflections:
                if reflection.reflection_type not in reflection_types:
                    reflection_types[reflection.reflection_type] = []
                reflection_types[reflection.reflection_type].append(reflection)
                
            # Generate insights
            for rtype, reflections in reflection_types.items():
                if len(reflections) >= 2:
                    insights[rtype] = {
                        "frequency": len(reflections),
                        "suggestions": {
                            "focus": rtype,
                            "action": "optimize",
                            "priority": len(reflections) / 10
                        }
                    }
                    
        # Analyze pattern frequency
        frequent_patterns = self.session.query(PatternMemory).filter(
            PatternMemory.frequency > 5
        ).all()
        
        if frequent_patterns:
            insights["frequent_patterns"] = {
                "count": len(frequent_patterns),
                "suggestions": {
                    "action": "create_templates",
                    "patterns": [p.pattern_content for p in frequent_patterns[:5]]
                }
            }
            
        return insights

# =====================================
# AURORA 7D PROCESSING ENGINE
# =====================================

class AuroraProcessor(QObject):
    """Aurora 7-Dimensionale Verarbeitungs-Engine mit erweitertem Speicher"""
    
    response_ready = pyqtSignal(str)
    dimension_updated = pyqtSignal(str, float)
    processing_complete = pyqtSignal(dict)
    status_update = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.ollama = OllamaProvider()
        self.memory = EnhancedMemorySystem()  # Using enhanced memory system
        
        # 7 Dimensions with detailed configuration
        self.dimensions = {
            'D0': {'value': 50, 'name': 'Cognitive', 'color': '#ff6b6b'},
            'D1': {'value': 50, 'name': 'Emotional', 'color': '#4ecdc4'},
            'D2': {'value': 50, 'name': 'Creative', 'color': '#45b7d1'},
            'D3': {'value': 50, 'name': 'Memory', 'color': '#96ceb4'},
            'D4': {'value': 50, 'name': 'Intuition', 'color': '#ffeaa7'},
            'D5': {'value': 50, 'name': 'Meta', 'color': '#dfe6e9'},
            'D6': {'value': 50, 'name': 'Unified', 'color': '#a29bfe'}
        }
        
        self.processing_active = False
        self.conversation_context = []
        
    def process_message(self, message: str, use_7d: bool = True, temperature: float = 0.7):
        """Verarbeitet Nachricht durch Aurora 7D-System mit erweitertem Speicher"""
        self.processing_active = True
        self.status_update.emit("🎯 Starting 7D Processing...")
        logger.info(f"🎯 Processing message: {message[:100]}...")
        
        try:
            # Pre-Reflection Phase
            pre_reflection = self._pre_reflection(message)
            
            # Determine memory type and category
            memory_type, category = self._categorize_message(message, pre_reflection)
            
            # Store in enhanced memory with metadata
            importance = min(1.0, len(message) / 200 + 0.3)
            meta_data = {
                "pre_reflection": pre_reflection,
                "dimensions": {k: v['value'] for k, v in self.dimensions.items()}
            }
            
            memory_id = self.memory.store_memory(
                content=message,
                memory_type=memory_type,
                category=category,
                importance=importance,
                metadata=meta_data
            )
            
            # 7D Processing if enabled
            if use_7d:
                dimension_outputs = self._process_dimensions(message, pre_reflection)
            else:
                dimension_outputs = {}
                
            # Build context from enhanced memory
            context = self._build_enhanced_context(message)
            
            # Generate response with Ollama
            self.status_update.emit("🤖 Generating with Ollama...")
            response = self._generate_response(message, pre_reflection, dimension_outputs, context, temperature)
            
            # Store response with learning metadata
            response_meta_data = {
                "generated_by": self.ollama.selected_model,
                "temperature": temperature,
                "context_used": len(context) > 0
            }
            
            self.memory.store_memory(
                content=response,
                memory_type="semantic",
                category="assistant_response",
                importance=importance * 0.8,
                metadata=response_meta_data
            )
            
            # Update knowledge graph if new concepts detected
            self._update_knowledge_from_interaction(message, response)
            
            # Update conversation context
            self.conversation_context.append({"role": "user", "content": message})
            self.conversation_context.append({"role": "assistant", "content": response})
            
            # Keep context manageable
            if len(self.conversation_context) > 20:
                self.conversation_context = self.conversation_context[-20:]
                
            # Emit results
            self.response_ready.emit(response)
            self.processing_complete.emit({
                "message": message,
                "response": response,
                "memory_id": memory_id,
                "memory_type": memory_type,
                "category": category,
                "dimensions": {k: v['value'] for k, v in self.dimensions.items()},
                "pre_reflection": pre_reflection,
                "context_size": len(context)
            })
            
            self.status_update.emit("✅ Processing complete")
            
        except Exception as e:
            error_msg = f"Error in processing: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            self.response_ready.emit(f"⚠️ {error_msg}")
            self.status_update.emit(f"❌ {error_msg}")
            
        finally:
            self.processing_active = False
            
    def _categorize_message(self, message: str, pre_reflection: Dict) -> Tuple[str, str]:
        """Kategorisiert die Nachricht für den Speicher"""
        intent = pre_reflection.get("intent", "statement")
        domain = pre_reflection.get("domain", "general")
        emotional_valence = pre_reflection.get("emotional_valence", 0.5)
        
        # Determine memory type
        if intent == "question":
            memory_type = "episodic"
        elif domain == "emotional" or emotional_valence < 0.3 or emotional_valence > 0.7:
            memory_type = "emotional"
        elif domain == "creative":
            memory_type = "procedural"
        elif intent == "request_info":
            memory_type = "semantic"
        else:
            memory_type = "contextual"
            
        # Determine category
        category = f"{domain}_{intent}"
        
        return memory_type, category
        
    def _build_enhanced_context(self, message: str) -> str:
        """Baut erweiterten Kontext aus SQL-Memory"""
        context_parts = []
        
        # Get semantically similar memories
        memories = self.memory.semantic_search(message, 5)
        if memories:
            context_parts.append("Relevant memories:")
            for mem_result in memories:
                memory = mem_result["memory"]
                similarity = mem_result["similarity"]
                content = memory.content[:100]
                mem_type = memory.memory_type
                context_parts.append(f"- [{similarity:.2f}|{mem_type}] {content}...")
                
        # Get associated memories
        if memories and memories[0]["memory"].associations:
            context_parts.append("\nAssociated contexts:")
            for assoc in memories[0]["memory"].associations[:3]:
                target_memory = assoc.target
                if target_memory:
                    context_parts.append(f"- [{assoc.association_type}] {target_memory.content[:50]}...")
                    
        # Add recent conversation
        if self.conversation_context:
            context_parts.append("\nRecent conversation:")
            for entry in self.conversation_context[-4:]:
                role = entry["role"]
                content = entry["content"][:100]
                context_parts.append(f"{role}: {content}...")
                
        # Add relevant meta-reflections
        reflections = self.memory.session.query(MetaReflection).filter(
            MetaReflection.applied == False
        ).order_by(MetaReflection.created_at.desc()).limit(2).all()
        
        if reflections:
            context_parts.append("\nActive learning insights:")
            for reflection in reflections:
                context_parts.append(f"- {reflection.content}")
                
        return "\n".join(context_parts)
        
    def _update_knowledge_from_interaction(self, message: str, response: str):
        """Extrahiert und speichert neues Wissen"""
        # Simple concept extraction (could be enhanced with NLP)
        words = (message + " " + response).split()
        
        # Look for definitions or explanations
        if "ist" in words or "bedeutet" in words or "heißt" in words:
            # Try to extract concept-definition pairs
            # This is simplified - could use more sophisticated NLP
            for i, word in enumerate(words):
                if word in ["ist", "bedeutet", "heißt"] and i > 0 and i < len(words) - 1:
                    concept = words[i-1].lower()
                    definition_start = i + 1
                    definition_words = words[definition_start:min(definition_start + 10, len(words))]
                    definition = " ".join(definition_words)
                    
                    if len(concept) > 2 and len(definition) > 10:
                        self.memory.update_knowledge_graph(
                            concept=concept,
                            definition=definition,
                            category="learned",
                            related_memories=[message[:50]]
                        )
                        
    def _pre_reflection(self, message: str) -> Dict:
        """Aurora Pre-Reflection Analyse"""
        return {
            "emotional_valence": self._analyze_emotion(message),
            "cognitive_complexity": min(1.0, len(message.split()) / 50),
            "intent": self._detect_intent(message),
            "context_required": self._check_context_requirement(message),
            "urgency": self._detect_urgency(message),
            "domain": self._detect_domain(message)
        }
        
    def _analyze_emotion(self, text: str) -> float:
        """Analysiert emotionale Valenz des Texts"""
        positive_words = ["gut", "freude", "glücklich", "super", "toll", "schön", "liebe", "danke", "ja"]
        negative_words = ["schlecht", "traurig", "wütend", "ärger", "problem", "schwierig", "nein", "nicht"]
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.5
            
        return pos_count / total
        
    def _detect_intent(self, text: str) -> str:
        """Erkennt Benutzerintention"""
        text_lower = text.lower()
        
        if "?" in text:
            return "question"
        elif any(word in text_lower for word in ["hilf", "zeig", "erkläre", "wie", "was ist"]):
            return "request_info"
        elif any(word in text_lower for word in ["mach", "erstelle", "generiere", "schreibe"]):
            return "request_action"
        elif any(word in text_lower for word in ["ich fühle", "ich denke", "mir geht", "ich bin"]):
            return "personal_statement"
        elif any(word in text_lower for word in ["danke", "gut gemacht", "toll"]):
            return "feedback_positive"
        else:
            return "statement"
            
    def _detect_urgency(self, text: str) -> float:
        """Erkennt Dringlichkeit der Anfrage"""
        urgent_words = ["sofort", "schnell", "dringend", "wichtig", "jetzt", "bitte"]
        text_lower = text.lower()
        urgent_count = sum(1 for word in urgent_words if word in text_lower)
        return min(1.0, urgent_count * 0.3)
        
    def _detect_domain(self, text: str) -> str:
        """Erkennt Themenbereich"""
        domains = {
            "technical": ["code", "programm", "bug", "error", "function", "system"],
            "emotional": ["fühle", "emotion", "gefühl", "stimmung", "laune"],
            "creative": ["idee", "kreativ", "erstelle", "design", "kunst"],
            "analytical": ["analyse", "daten", "statistik", "berechne", "logik"]
        }
        
        text_lower = text.lower()
        for domain, keywords in domains.items():
            if any(keyword in text_lower for keyword in keywords):
                return domain
                
        return "general"
        
    def _check_context_requirement(self, text: str) -> bool:
        """Prüft ob Kontext aus vorherigen Nachrichten benötigt wird"""
        context_indicators = ["das", "dies", "jenes", "vorhin", "eben", "letzte", "davon", "darüber"]
        return any(word in text.lower() for word in context_indicators)
        
    def _process_dimensions(self, message: str, pre_reflection: Dict) -> Dict:
        """Verarbeitet Nachricht durch alle 7 Aurora-Dimensionen"""
        outputs = {}
        
        for dim_id, dim_info in self.dimensions.items():
            weight = dim_info['value'] / 100.0
            name = dim_info['name']
            
            if dim_id == "D0":  # Cognitive
                analysis = self._cognitive_analysis(message)
                outputs[dim_id] = f"{name} (w:{weight:.2f}): {analysis}"
                
            elif dim_id == "D1":  # Emotional
                valence = pre_reflection["emotional_valence"]
                emotion_type = "positive" if valence > 0.6 else "negative" if valence < 0.4 else "neutral"
                outputs[dim_id] = f"{name} (w:{weight:.2f}): Emotional resonance detected - {emotion_type} (v:{valence:.2f})"
                
            elif dim_id == "D2":  # Creative
                creativity_score = self._assess_creativity_potential(message)
                outputs[dim_id] = f"{name} (w:{weight:.2f}): Creative potential: {creativity_score:.2f}"
                
            elif dim_id == "D3":  # Memory
                memories = self.memory.semantic_search(message, 3)
                memory_info = f"{len(memories)} relevant memories found"
                if memories:
                    top_similarity = memories[0]["similarity"]
                    memory_info += f" (top similarity: {top_similarity:.2f})"
                outputs[dim_id] = f"{name} (w:{weight:.2f}): {memory_info}"
                
            elif dim_id == "D4":  # Intuition
                intuition = self._intuitive_assessment(message, pre_reflection)
                outputs[dim_id] = f"{name} (w:{weight:.2f}): {intuition}"
                
            elif dim_id == "D5":  # Meta
                meta = self._meta_cognitive_analysis(message, pre_reflection)
                outputs[dim_id] = f"{name} (w:{weight:.2f}): {meta}"
                
            elif dim_id == "D6":  # Unified
                unity_score = self._calculate_dimensional_unity()
                outputs[dim_id] = f"{name} (w:{weight:.2f}): Dimensional unity: {unity_score:.2f}"
                
            # Update dimension visualization
            self.dimension_updated.emit(dim_id, weight * 100)
            
        return outputs
        
    def _cognitive_analysis(self, message: str) -> str:
        """Kognitive Analyse der Nachricht"""
        words = message.split()
        complexity = len(words)
        
        if complexity < 10:
            return "Simple query - direct processing"
        elif complexity < 30:
            return "Moderate complexity - structured analysis"
        else:
            return "Complex query - deep analytical processing required"
            
    def _assess_creativity_potential(self, message: str) -> float:
        """Bewertet kreatives Potenzial"""
        creative_indicators = ["idee", "neu", "anders", "kreativ", "vorstellen", "was wäre wenn"]
        score = sum(1 for indicator in creative_indicators if indicator in message.lower())
        return min(1.0, score * 0.3)
        
    def _intuitive_assessment(self, message: str, pre_reflection: Dict) -> str:
        """Intuitive Einschätzung basierend auf Mustern"""
        if pre_reflection["urgency"] > 0.7:
            return "High urgency detected - prioritizing response"
        elif pre_reflection["emotional_valence"] < 0.3:
            return "Emotional support may be needed"
        elif pre_reflection["domain"] == "creative":
            return "Creative exploration suggested"
        else:
            return "Standard processing pattern appropriate"
            
    def _meta_cognitive_analysis(self, message: str, pre_reflection: Dict) -> str:
        """Meta-kognitive Analyse des Verarbeitungsprozesses"""
        if pre_reflection["cognitive_complexity"] > 0.7:
            return "Complex reasoning required - engaging deep analysis"
        elif pre_reflection["context_required"]:
            return "Context integration necessary - accessing memory"
        else:
            return "Direct response pattern selected"
            
    def _calculate_dimensional_unity(self) -> float:
        """Berechnet Harmonie zwischen Dimensionen"""
        values = [dim['value'] for dim in self.dimensions.values()]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        # Lower variance = higher unity
        unity = 1.0 - min(1.0, variance / 1000)
        return unity
        
    def _generate_response(self, message: str, pre_reflection: Dict, 
                          dimensions: Dict, context: str, temperature: float) -> str:
        """Generiert finale Antwort mit Ollama unter Berücksichtigung aller Faktoren"""
        if not self.ollama.available:
            return self._fallback_response(message, pre_reflection, dimensions)
            
        # Build Aurora-specific prompt
        prompt = self._build_aurora_prompt(message, pre_reflection, dimensions, context)
        
        # Generate with Ollama
        response = self.ollama.generate(
            prompt=prompt,
            model=self.ollama.selected_model,
            temperature=temperature,
            max_tokens=2000
        )
        
        return response
        
    def _build_aurora_prompt(self, message: str, pre_reflection: Dict, 
                            dimensions: Dict, context: str) -> str:
        """Baut detaillierten Aurora-Prompt für Ollama"""
        
        # Build system description
        prompt_parts = [
            "You are Aurora, a 7-dimensional consciousness system with cognitive-emotional processing capabilities and persistent memory.",
            "You have access to a long-term memory system that stores and learns from all interactions.",
            "Respond naturally and helpfully while maintaining awareness of the dimensional processing and memory context.",
            ""
        ]
        
        # Add pre-reflection insights
        prompt_parts.append("Pre-Reflection Analysis:")
        prompt_parts.append(f"- Emotional Valence: {pre_reflection['emotional_valence']:.2f}")
        prompt_parts.append(f"- Intent: {pre_reflection['intent']}")
        prompt_parts.append(f"- Domain: {pre_reflection['domain']}")
        prompt_parts.append(f"- Urgency: {pre_reflection['urgency']:.2f}")
        prompt_parts.append(f"- Context Required: {pre_reflection['context_required']}")
        prompt_parts.append("")
        
        # Add dimensional processing if available
        if dimensions:
            prompt_parts.append("7D Processing Results:")
            for dim_output in dimensions.values():
                prompt_parts.append(f"- {dim_output}")
            prompt_parts.append("")
            
        # Add context if available
        if context:
            prompt_parts.append("Memory Context and Previous Knowledge:")
            prompt_parts.append(context)
            prompt_parts.append("")
            
        # Add the actual message
        prompt_parts.append(f"User Message: {message}")
        prompt_parts.append("")
        prompt_parts.append("Aurora Response:")
        
        return "\n".join(prompt_parts)
        
    def _fallback_response(self, message: str, pre_reflection: Dict, dimensions: Dict) -> str:
        """Fallback-Antwort wenn Ollama nicht verfügbar"""
        intent = pre_reflection["intent"]
        domain = pre_reflection["domain"]
        
        base_responses = {
            "question": "Based on my 7D analysis and memory search, I understand you're asking a question. While Ollama is offline, I can tell you that ",
            "request_info": "I'd help you with that information. My dimensional processing and memory system suggests ",
            "request_action": "I understand you want me to create something. Without Ollama, I can outline that ",
            "personal_statement": "Thank you for sharing that with me. My emotional dimension and memory patterns resonate with ",
            "feedback_positive": "Thank you for the positive feedback! This helps me learn and improve. ",
            "statement": "I've processed your statement through all 7 dimensions and stored it in my memory. "
        }
        
        response = base_responses.get(intent, "Message processed and stored. ")
        
        # Add domain-specific info
        if domain == "technical":
            response += "From a technical perspective, this involves systematic analysis. "
        elif domain == "emotional":
            response += "I sense the emotional significance of this. "
        elif domain == "creative":
            response += "This opens creative possibilities to explore. "
            
        # Add dimension info
        active_dims = [info['name'] for dim_id, info in self.dimensions.items() 
                      if info['value'] > 60]
        if active_dims:
            response += f"[Active dimensions: {', '.join(active_dims)}]"
            
        # Add memory info
        memories = self.memory.semantic_search(message, 2)
        if memories:
            response += f" [Found {len(memories)} related memories in my database]"
            
        return response
        
    def update_dimension(self, dimension: str, value: int):
        """Aktualisiert Dimensionswert"""
        if dimension in self.dimensions:
            self.dimensions[dimension]['value'] = value
            logger.info(f"📊 Dimension {dimension} updated to {value}")
            
    def get_dimension_info(self) -> Dict:
        """Gibt Dimensions-Informationen zurück"""
        return self.dimensions.copy()

# =====================================
# UI COMPONENTS
# =====================================

class DimensionSlider(QWidget):
    """Aurora Dimension Control Slider"""
    
    valueChanged = pyqtSignal(str, int)
    
    def __init__(self, dimension: str, color: str, parent=None):
        super().__init__(parent)
        self.dimension = dimension
        self.color = color
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(4)
        
        # Header
        header_layout = QHBoxLayout()
        self.label = QLabel(self.dimension)
        self.label.setStyleSheet(f"color: {self.color}; font-weight: 600;")
        self.value_label = QLabel("50")
        self.value_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        header_layout.addWidget(self.label)
        header_layout.addStretch()
        header_layout.addWidget(self.value_label)
        
        # Slider with Aurora styling
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(self.on_value_changed)
        
        self.slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background: {COLORS['bg_tertiary']};
                height: 6px;
                border-radius: 3px;
            }}
            QSlider::sub-page:horizontal {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {self.color}88, stop:1 {self.color});
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {self.color};
                width: 16px;
                height: 16px;
                border-radius: 8px;
                margin: -5px 0;
                border: 2px solid {COLORS['bg_primary']};
            }}
            QSlider::handle:horizontal:hover {{
                width: 18px;
                height: 18px;
                margin: -6px 0;
            }}
        """)
        
        layout.addLayout(header_layout)
        layout.addWidget(self.slider)
        self.setLayout(layout)
        
    def on_value_changed(self, value):
        self.value_label.setText(str(value))
        self.valueChanged.emit(self.dimension, value)

class MemoryAllocationBar(QWidget):
    """Memory Allocation Visualization"""
    
    def __init__(self, label: str, color: str, parent=None):
        super().__init__(parent)
        self.label = label
        self.color = color
        self.value = 0
        self.max_value = 100
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(4)
        
        # Header
        header_layout = QHBoxLayout()
        label_widget = QLabel(self.label)
        label_widget.setStyleSheet(f"color: {COLORS['text_primary']}; font-weight: 600;")
        
        self.value_label = QLabel("0 MB")
        self.value_label.setStyleSheet(f"color: {self.color}; font-weight: bold;")
        
        header_layout.addWidget(label_widget)
        header_layout.addStretch()
        header_layout.addWidget(self.value_label)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setMaximum(self.max_value)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(20)
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                background-color: {COLORS['bg_tertiary']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {self.color}88, stop:1 {self.color});
                border-radius: 3px;
            }}
        """)
        
        layout.addLayout(header_layout)
        layout.addWidget(self.progress)
        self.setLayout(layout)
        
    def set_value(self, value: int, text: str = None):
        self.value = value
        self.progress.setValue(value)
        if text:
            self.value_label.setText(text)

class PluginCard(QFrame):
    """Plugin Status Card"""
    
    statusChanged = pyqtSignal(str, bool)
    
    def __init__(self, plugin_name: str, parent=None):
        super().__init__(parent)
        self.plugin_name = plugin_name
        self.is_running = False
        self.setup_ui()
        
    def setup_ui(self):
        """Setup UI with proper initialization"""
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 12px;
            }}
        """)
        
        layout = QVBoxLayout()
        
        # Header
        header_layout = QHBoxLayout()
        self.name_label = QLabel(self.plugin_name)
        self.name_label.setStyleSheet(f"""
            font-size: 16px;
            font-weight: 600;
            color: {COLORS['accent_blue']};
        """)
        
        self.status_indicator = QLabel("●")
        
        header_layout.addWidget(self.name_label)
        header_layout.addStretch()
        header_layout.addWidget(self.status_indicator)
        
        # Status text
        self.status_text = QLabel("Stopped")
        self.status_text.setStyleSheet(f"color: {COLORS['text_secondary']};")
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent_green']};
                color: white;
                font-weight: 600;
                padding: 6px 12px;
                border-radius: 4px;
                border: none;
            }}
            QPushButton:hover {{
                background-color: #2ea043;
            }}
        """)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent_red']};
                color: white;
                font-weight: 600;
                padding: 6px 12px;
                border-radius: 4px;
                border: none;
            }}
            QPushButton:hover {{
                background-color: #da3633;
            }}
        """)
        self.stop_btn.setEnabled(False)
        
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        
        # Assemble layout
        layout.addLayout(header_layout)
        layout.addWidget(self.status_text)
        layout.addWidget(self.progress)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        # Connect signals
        self.start_btn.clicked.connect(self.start_plugin)
        self.stop_btn.clicked.connect(self.stop_plugin)
        
        # Initial status
        self.update_status(False)
        
    def update_status(self, running: bool):
        """Update plugin status"""
        self.is_running = running
        if running:
            self.status_indicator.setStyleSheet(f"color: {COLORS['accent_green']}; font-size: 20px;")
            self.status_text.setText("Running")
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.progress.setVisible(True)
            self.progress.setRange(0, 0)
        else:
            self.status_indicator.setStyleSheet(f"color: {COLORS['accent_red']}; font-size: 20px;")
            self.status_text.setText("Stopped")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.progress.setVisible(False)
            
    def start_plugin(self):
        """Start plugin"""
        self.update_status(True)
        self.statusChanged.emit(self.plugin_name, True)
        logger.info(f"✅ {self.plugin_name} started")
        
    def stop_plugin(self):
        """Stop plugin"""
        self.update_status(False)
        self.statusChanged.emit(self.plugin_name, False)
        logger.info(f"ℹ️ {self.plugin_name} stopped")

# =====================================
# MAIN AURORA WINDOW
# =====================================

class AuroraCompleteWindow(QMainWindow):
    """Aurora Complete System Main Window with Full Ollama Integration and Enhanced Memory"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize core systems
        self.processor = AuroraProcessor()
        
        # Initialize UI elements
        self.init_ui_elements()
        
        # Setup everything
        self.setup_ui()
        self.setup_connections()
        self.setup_timers()
        self.initialize_system()
        
    def init_ui_elements(self):
        """Initialize all UI element references"""
        self.ollama_status = None
        self.model_combo = None
        self.chat_display = None
        self.input_field = None
        self.send_btn = None
        self.use_7d_check = None
        self.temp_slider = None
        self.temp_label = None
        self.dimension_sliders = {}
        self.processing_status = None
        self.memory_stats_labels = {}
        self.stm_bar = None
        self.ltm_bar = None
        self.memory_search = None
        self.search_btn = None
        self.search_results = None
        self.console = None
        self.perf_labels = {}
        self.status_label = None
        self.time_label = None
        self.llm_card = None
        self.memory_card = None
        self.processor_card = None
        self.init_btn = None
        self.consolidate_btn = None
        self.export_btn = None
        self.pull_model_btn = None
        
    def setup_ui(self):
        """Setup complete UI"""
        self.setWindowTitle("🌌 Aurora Complete System - Enhanced Memory & Self-Development")
        self.setGeometry(100, 100, 1400, 900)
        
        # Apply Aurora dark theme
        self.setStyleSheet(self.get_stylesheet())
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(16, 16, 16, 16)
        
        # Header
        header = self.create_header()
        main_layout.addWidget(header)
        
        # Content splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Panels
        left_panel = self.create_left_panel()
        center_panel = self.create_center_panel()
        right_panel = self.create_right_panel()
        
        splitter.addWidget(left_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(right_panel)
        
        splitter.setSizes([350, 700, 350])
        main_layout.addWidget(splitter)
        
        # Status bar
        self.create_status_bar()
        
        central_widget.setLayout(main_layout)
        
    def create_header(self):
        """Create header with Ollama status"""
        header = QWidget()
        header.setMaximumHeight(80)
        header.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['bg_secondary']};
                border-radius: 8px;
                border: 1px solid {COLORS['border']};
            }}
        """)
        
        layout = QHBoxLayout()
        
        # Title
        title_label = QLabel("🌌 Aurora Complete System")
        title_label.setStyleSheet(f"""
            font-size: 24px;
            font-weight: 700;
            color: {COLORS['accent_purple']};
        """)
        
        # Ollama status
        self.ollama_status = QLabel("● Ollama: Checking...")
        self.ollama_status.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 14px;")
        
        # Model selector
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(200)
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        
        # Pull model button
        self.pull_model_btn = QPushButton("📥 Pull Model")
        self.pull_model_btn.clicked.connect(self.pull_new_model)
        
        layout.addWidget(title_label)
        layout.addStretch()
        layout.addWidget(self.ollama_status)
        layout.addWidget(self.model_combo)
        layout.addWidget(self.pull_model_btn)
        
        header.setLayout(layout)
        return header
        
    def create_left_panel(self):
        """Create left control panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Plugin Management
        plugin_group = QGroupBox("Plugin Management")
        plugin_layout = QVBoxLayout()
        
        self.llm_card = PluginCard("LLM Plugin (Ollama)")
        self.memory_card = PluginCard("Enhanced Memory (SQL)")
        self.processor_card = PluginCard("7D Processor")
        
        plugin_layout.addWidget(self.llm_card)
        plugin_layout.addWidget(self.memory_card)
        plugin_layout.addWidget(self.processor_card)
        
        plugin_group.setLayout(plugin_layout)
        
        # System Controls
        system_group = QGroupBox("System Control")
        system_layout = QVBoxLayout()
        
        self.init_btn = QPushButton("Initialize All")
        self.init_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLORS['gradient_1']}, stop:1 {COLORS['gradient_2']});
                color: white;
                font-weight: 600;
                padding: 10px;
                border-radius: 6px;
            }}
        """)
        
        self.consolidate_btn = QPushButton("Consolidate & Learn")
        self.export_btn = QPushButton("Export Knowledge")
        
        system_layout.addWidget(self.init_btn)
        system_layout.addWidget(self.consolidate_btn)
        system_layout.addWidget(self.export_btn)
        
        system_group.setLayout(system_layout)
        
        layout.addWidget(plugin_group)
        layout.addWidget(system_group)
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel
        
    def create_center_panel(self):
        """Create center interaction panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Tabs
        tabs = QTabWidget()
        
        # Chat tab
        chat_tab = self.create_chat_tab()
        tabs.addTab(chat_tab, "💬 Chat")
        
        # Dimensions tab
        dimensions_tab = self.create_dimensions_tab()
        tabs.addTab(dimensions_tab, "🎨 Dimensions")
        
        # Memory tab
        memory_tab = self.create_memory_tab()
        tabs.addTab(memory_tab, "🧠 Memory")
        
        layout.addWidget(tabs)
        panel.setLayout(layout)
        return panel
        
    def create_chat_tab(self):
        """Create chat interface"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Chat display
        self.chat_display = QTextBrowser()
        self.chat_display.setMinimumHeight(400)
        self.chat_display.setHtml(f"""
            <div style='color: #8b949e;'>
                <h3 style='color: #a371f7;'>Aurora System Ready</h3>
                <p>Enhanced Memory System Active - SQL Database: {AURORA_STORAGE_PATH}</p>
                <p style='color: #58a6ff;'>Self-Development Mode: Enabled</p>
                <p>Waiting for Ollama connection...</p>
            </div>
        """)
        
        # Input area
        input_group = QGroupBox("Message Input")
        input_layout = QVBoxLayout()
        
        self.input_field = QTextEdit()
        self.input_field.setMaximumHeight(100)
        self.input_field.setPlaceholderText("Enter your message...")
        
        # Controls
        control_layout = QHBoxLayout()
        
        self.send_btn = QPushButton("Send")
        self.send_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent_blue']};
                color: white;
                font-weight: 600;
                padding: 8px 20px;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: #1f6feb;
            }}
        """)
        
        self.use_7d_check = QCheckBox("Use 7D Processing")
        self.use_7d_check.setChecked(True)
        
        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(0, 100)
        self.temp_slider.setValue(70)
        self.temp_slider.setMaximumWidth(150)
        
        self.temp_label = QLabel("Temp: 0.7")
        
        control_layout.addWidget(self.send_btn)
        control_layout.addWidget(self.use_7d_check)
        control_layout.addStretch()
        control_layout.addWidget(QLabel("Temperature:"))
        control_layout.addWidget(self.temp_slider)
        control_layout.addWidget(self.temp_label)
        
        input_layout.addWidget(self.input_field)
        input_layout.addLayout(control_layout)
        input_group.setLayout(input_layout)
        
        layout.addWidget(self.chat_display)
        layout.addWidget(input_group)
        
        widget.setLayout(layout)
        return widget
        
    def create_dimensions_tab(self):
        """Create dimensions control tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("7-Dimensional Processing Control")
        title.setStyleSheet(f"""
            font-size: 16px;
            font-weight: 600;
            color: {COLORS['accent_purple']};
            padding: 8px;
        """)
        layout.addWidget(title)
        
        # Dimension sliders
        dimensions_group = QGroupBox("Dimension Weights")
        dim_layout = QVBoxLayout()
        
        for dim, color in DIMENSION_COLORS.items():
            slider = DimensionSlider(dim, color)
            slider.valueChanged.connect(self.on_dimension_changed)
            self.dimension_sliders[dim] = slider
            dim_layout.addWidget(slider)
            
        dimensions_group.setLayout(dim_layout)
        
        # Processing status
        status_group = QGroupBox("Processing Status")
        status_layout = QVBoxLayout()
        
        self.processing_status = QTextBrowser()
        self.processing_status.setMaximumHeight(150)
        
        status_layout.addWidget(self.processing_status)
        status_group.setLayout(status_layout)
        
        layout.addWidget(dimensions_group)
        layout.addWidget(status_group)
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget
        
    def create_memory_tab(self):
        """Create memory management tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Memory stats
        stats_group = QGroupBox("Memory Statistics")
        stats_layout = QGridLayout()
        
        self.memory_stats_labels = {}
        stats = [
            "Total Memories", "Memory Types", "Patterns", "Associations",
            "Reflections", "Knowledge Nodes", "DB Size (MB)", "Storage Path"
        ]
        
        for i, stat in enumerate(stats):
            label = QLabel(f"{stat}:")
            value = QLabel("0")
            value.setStyleSheet(f"color: {COLORS['accent_cyan']}; font-weight: 600;")
            self.memory_stats_labels[stat] = value
            stats_layout.addWidget(label, i // 2, (i % 2) * 2)
            stats_layout.addWidget(value, i // 2, (i % 2) * 2 + 1)
            
        stats_group.setLayout(stats_layout)
        
        # Memory allocation bars
        self.stm_bar = MemoryAllocationBar("STM Allocation", COLORS['accent_cyan'])
        self.ltm_bar = MemoryAllocationBar("LTM Allocation", COLORS['accent_blue'])
        
        # Search
        search_group = QGroupBox("Semantic Search")
        search_layout = QVBoxLayout()
        
        search_input_layout = QHBoxLayout()
        self.memory_search = QLineEdit()
        self.memory_search.setPlaceholderText("Search memories...")
        self.search_btn = QPushButton("Search")
        search_input_layout.addWidget(self.memory_search)
        search_input_layout.addWidget(self.search_btn)
        
        self.search_results = QListWidget()
        self.search_results.setMaximumHeight(200)
        
        search_layout.addLayout(search_input_layout)
        search_layout.addWidget(self.search_results)
        search_group.setLayout(search_layout)
        
        layout.addWidget(stats_group)
        layout.addWidget(self.stm_bar)
        layout.addWidget(self.ltm_bar)
        layout.addWidget(search_group)
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget
        
    def create_right_panel(self):
        """Create right analytics panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Console
        console_group = QGroupBox("System Console")
        console_layout = QVBoxLayout()
        
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet(f"""
            QTextEdit {{
                background-color: #000000;
                color: #00ff00;
                font-family: 'Cascadia Code', 'Courier New', monospace;
                font-size: 11px;
            }}
        """)
        
        console_layout.addWidget(self.console)
        console_group.setLayout(console_layout)
        
        # Performance
        perf_group = QGroupBox("Performance Metrics")
        perf_layout = QGridLayout()
        
        self.perf_labels = {}
        metrics = ["Response Time", "Token Count", "Memory Usage", "Cache Hits"]
        
        for i, metric in enumerate(metrics):
            label = QLabel(f"{metric}:")
            value = QLabel("0")
            value.setStyleSheet(f"color: {COLORS['accent_green']};")
            self.perf_labels[metric] = value
            perf_layout.addWidget(label, i, 0)
            perf_layout.addWidget(value, i, 1)
            
        perf_group.setLayout(perf_layout)
        
        layout.addWidget(console_group)
        layout.addWidget(perf_group)
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel
        
    def create_status_bar(self):
        """Create status bar"""
        status_bar = self.statusBar()
        status_bar.setStyleSheet(f"""
            QStatusBar {{
                background-color: {COLORS['bg_secondary']};
                color: {COLORS['text_secondary']};
                border-top: 1px solid {COLORS['border']};
            }}
        """)
        
        self.status_label = QLabel("System Ready")
        self.time_label = QLabel("")
        
        status_bar.addPermanentWidget(self.time_label)
        status_bar.showMessage(f"Aurora System - Enhanced Memory at: {AURORA_STORAGE_PATH}")
        
    def setup_connections(self):
        """Setup signal connections"""
        # Buttons
        self.send_btn.clicked.connect(self.send_message)
        self.init_btn.clicked.connect(self.initialize_all)
        self.consolidate_btn.clicked.connect(self.consolidate_memory)
        self.export_btn.clicked.connect(self.export_knowledge)
        self.search_btn.clicked.connect(self.search_memory)
        self.temp_slider.valueChanged.connect(self.update_temperature)
        
        # Processor signals
        self.processor.response_ready.connect(self.display_response)
        self.processor.processing_complete.connect(self.on_processing_complete)
        self.processor.status_update.connect(self.update_console_status)
        
        # Plugin cards
        self.llm_card.statusChanged.connect(self.on_plugin_status_changed)
        self.memory_card.statusChanged.connect(self.on_plugin_status_changed)
        self.processor_card.statusChanged.connect(self.on_plugin_status_changed)
        
    def setup_timers(self):
        """Setup update timers"""
        # Status timer
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)
        
        # Memory timer
        self.memory_timer = QTimer(self)
        self.memory_timer.timeout.connect(self.update_memory_stats)
        self.memory_timer.start(5000)
        
        # Ollama check timer
        self.ollama_timer = QTimer(self)
        self.ollama_timer.timeout.connect(self.check_ollama_status)
        self.ollama_timer.start(10000)  # Check every 10 seconds
        
    def initialize_system(self):
        """Initialize system on startup"""
        self.log_console("🚀 Aurora Complete System starting...")
        self.log_console(f"📁 Memory storage: {AURORA_STORAGE_PATH}")
        self.log_console(f"🗄️ Database: {DATABASE_PATH}")
        self.check_ollama_status()
        self.update_memory_stats()
        self.log_console("✅ System initialized with enhanced memory")
        
    def check_ollama_status(self):
        """Check Ollama status and update UI"""
        if self.processor.ollama.check_connection():
            self.ollama_status.setText("● Ollama: Online")
            self.ollama_status.setStyleSheet(f"color: {COLORS['accent_green']}; font-size: 14px;")
            
            # Update model combo
            models = self.processor.ollama.models
            current_model = self.model_combo.currentText()
            
            self.model_combo.clear()
            if models:
                self.model_combo.addItems(models)
                # Restore selection if possible
                if current_model in models:
                    self.model_combo.setCurrentText(current_model)
                else:
                    self.processor.ollama.set_model(self.model_combo.itemText(0))

                self.log_console(f"✅ Ollama online with {len(models)} models: {', '.join(models)}")
            else:
                self.log_console("⚠️ Ollama online but no models found")
                self.model_combo.addItem("No models - Pull a model")
        else:
            self.ollama_status.setText("● Ollama: Offline")
            self.ollama_status.setStyleSheet(f"color: {COLORS['accent_red']}; font-size: 14px;")
            self.log_console("⚠️ Ollama not available - start 'ollama serve'")
            
    def on_model_changed(self, model_name: str):
        """Handle model selection change"""
        if model_name and self.processor.ollama.set_model(model_name):
            self.log_console(f"🔄 Model changed to: {model_name}")
            
    def pull_new_model(self):
        """Pull a new model from Ollama"""
        model_name, ok = QInputDialog.getText(
            self, "Pull Model", 
            "Enter model name (e.g., mistral, llama3, codellama):"
        )
        
        if ok and model_name:
            self.log_console(f"📥 Pulling model: {model_name}")
            self.pull_model_btn.setEnabled(False)
            
            # Run in thread
            def pull():
                success = self.processor.ollama.pull_model(model_name)
                # Use a signal to update UI from the main thread
                QTimer.singleShot(0, lambda: self.on_pull_complete(model_name, success))
                
            threading.Thread(target=pull, daemon=True).start()

    def on_pull_complete(self, model_name, success):
        """Handles UI updates after a model pull is finished."""
        if success:
            self.log_console(f"✅ Model {model_name} pulled successfully")
            self.check_ollama_status()
        else:
            self.log_console(f"❌ Failed to pull model {model_name}")
        self.pull_model_btn.setEnabled(True)
            
    def initialize_all(self):
        """Initialize all plugins"""
        self.log_console("🔧 Initializing all plugins...")
        
        # Start all plugins
        self.llm_card.start_plugin()
        self.memory_card.start_plugin()
        self.processor_card.start_plugin()
        
        # Check Ollama again
        self.check_ollama_status()
        
        self.log_console("✅ All plugins initialized")
        
    def send_message(self):
        """Send message for processing"""
        message = self.input_field.toPlainText().strip()
        if not message or self.processor.processing_active:
            return
            
        # Display user message
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.chat_display.append(f"\n<b style='color: {COLORS['accent_blue']}'>[{timestamp}] You:</b> {message}")
        
        # Clear input
        self.input_field.clear()
        
        # Disable send button during processing
        self.send_btn.setEnabled(False)
        self.send_btn.setText("Processing...")
        
        # Log
        self.log_console(f"📤 Processing: {message[:50]}...")
        
        # Get settings
        use_7d = self.use_7d_check.isChecked()
        temperature = self.temp_slider.value() / 100.0
        
        # Process in thread
        threading.Thread(
            target=lambda: self.processor.process_message(message, use_7d, temperature),
            daemon=True
        ).start()
        
    def display_response(self, response: str):
        """Display response in chat"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Format response with Aurora styling
        formatted_response = response.replace("\n", "<br>")
        self.chat_display.append(
            f"\n<b style='color: {COLORS['accent_purple']}'>[{timestamp}] Aurora:</b><br>"
            f"<span style='color: {COLORS['text_primary']}'>{formatted_response}</span>"
        )
        
        # Re-enable send button
        self.send_btn.setEnabled(True)
        self.send_btn.setText("Send")
        
        self.log_console(f"📥 Response generated ({len(response)} chars)")
        
    def on_processing_complete(self, data: Dict):
        """Handle processing completion"""
        # Update processing status
        status_html = "<div style='font-family: monospace; font-size: 11px;'>"
        status_html += f"<p>Processed at {datetime.now().strftime('%H:%M:%S')}</p>"
        
        if "memory_type" in data:
            status_html += f"<p>Memory Type: {data['memory_type']}</p>"
            status_html += f"<p>Category: {data.get('category', 'unknown')}</p>"
            
        if "pre_reflection" in data:
            pr = data["pre_reflection"]
            status_html += f"<p>Emotional Valence: {pr.get('emotional_valence', 0):.2f}</p>"
            status_html += f"<p>Intent: {pr.get('intent', 'unknown')}</p>"
            status_html += f"<p>Domain: {pr.get('domain', 'general')}</p>"
            status_html += f"<p>Urgency: {pr.get('urgency', 0):.2f}</p>"
            
        if "context_size" in data:
            status_html += f"<p>Context Size: {data['context_size']} chars</p>"
            
        status_html += "</div>"
        self.processing_status.setHtml(status_html)
        
        # Update performance metrics
        if "memory_id" in data:
            self.perf_labels["Token Count"].setText(str(len(data.get("message", "").split())))
            
    def on_dimension_changed(self, dimension: str, value: int):
        """Handle dimension value change"""
        self.processor.update_dimension(dimension, value)
        self.log_console(f"📊 Dimension {dimension} set to {value}")
        
    def update_temperature(self, value: int):
        """Update temperature display"""
        temp = value / 100.0
        self.temp_label.setText(f"Temp: {temp:.2f}")
        
    def search_memory(self):
        """Search memories"""
        query = self.memory_search.text()
        if not query:
            return
            
        results = self.processor.memory.semantic_search(query, 10)
        self.search_results.clear()
        
        if not results:
            self.search_results.addItem("No relevant memories found.")
            
        for result in results:
            memory = result["memory"]
            similarity = result["similarity"]
            relevance = result["relevance"]
            mem_type = memory.memory_type
            item_text = f"[{mem_type}|S:{similarity:.2f}|R:{relevance:.2f}] {memory.content[:50]}..."
            self.search_results.addItem(item_text)
            
        self.log_console(f"🔍 Found {len(results)} memories for: {query}")
        
    def consolidate_memory(self):
        """Consolidate memories and trigger self-development"""
        self.log_console("🧠 Starting memory consolidation and self-development...")
        self.consolidate_btn.setEnabled(False)
        
        def consolidate():
            self.processor.memory.consolidate_memories()
            QTimer.singleShot(0, self.on_consolidate_complete)

        threading.Thread(target=consolidate, daemon=True).start()

    def on_consolidate_complete(self):
        self.update_memory_stats()
        self.consolidate_btn.setEnabled(True)
        self.log_console("✅ Consolidation and learning complete")
        
    def export_knowledge(self):
        """Export Aurora's knowledge"""
        self.log_console("📁 Exporting knowledge...")
        self.export_btn.setEnabled(False)
        
        def export():
            filepath = self.processor.memory.export_knowledge()
            QTimer.singleShot(0, lambda: self.on_export_complete(filepath))

        threading.Thread(target=export, daemon=True).start()

    def on_export_complete(self, filepath):
        self.log_console(f"✅ Knowledge exported to: {filepath}")
        self.export_btn.setEnabled(True)
        
    def update_memory_stats(self):
        """Update memory statistics display"""
        stats = self.processor.memory.get_statistics()
        
        self.memory_stats_labels["Total Memories"].setText(str(stats["total_memories"]))
        
        types_text = ", ".join([f"{k}: {v}" for k, v in stats.get("memory_types", {}).items()])
        self.memory_stats_labels["Memory Types"].setText(types_text[:50] + "..." if len(types_text) > 50 else types_text)
            
        self.memory_stats_labels["Patterns"].setText(str(stats.get("patterns_count", 0)))
        self.memory_stats_labels["Associations"].setText(str(stats.get("associations_count", 0)))
        self.memory_stats_labels["Reflections"].setText(str(stats.get("reflections_count", 0)))
        self.memory_stats_labels["Knowledge Nodes"].setText(str(stats.get("knowledge_nodes", 0)))
            
        db_size = stats.get("database_size_mb", 0)
        self.memory_stats_labels["DB Size (MB)"].setText(f"{db_size:.2f}")
            
        path = stats.get("storage_path", "Unknown")
        display_path = "..." + path[-30:] if len(path) > 30 else path
        self.memory_stats_labels["Storage Path"].setText(display_path)
        
        self.stm_bar.set_value(int(stats["stm_allocation_mb"]), f"{stats['stm_allocation_mb']} MB")
        self.ltm_bar.set_value(int(stats["ltm_allocation_mb"]), f"{stats['ltm_allocation_mb']} MB")
            
    def update_status(self):
        """Update status displays"""
        current_time = datetime.now().strftime("%H:%M:%S")
        self.time_label.setText(current_time)
            
    def update_console_status(self, message: str):
        """Update console with status message"""
        self.log_console(message)
        
    def on_plugin_status_changed(self, plugin_name: str, running: bool):
        """Handle plugin status change"""
        status = "started" if running else "stopped"
        self.log_console(f"🔌 {plugin_name} {status}")
        
    def log_console(self, message: str):
        """Log message to console"""
        if self.console:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.console.append(f"[{timestamp}] {message}")
            
            # Auto-scroll
            cursor = self.console.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self.console.setTextCursor(cursor)
            
    def get_stylesheet(self):
        """Get complete Aurora stylesheet"""
        return f"""
        QWidget {{
            background-color: {COLORS['bg_primary']};
            color: {COLORS['text_primary']};
            font-family: 'Segoe UI', -apple-system, sans-serif;
        }}
        
        QMainWindow {{
            background-color: {COLORS['bg_primary']};
        }}
        
        QGroupBox {{
            background-color: {COLORS['bg_secondary']};
            border: 1px solid {COLORS['border']};
            border-radius: 8px;
            margin-top: 12px;
            padding-top: 16px;
            font-weight: 600;
        }}
        
        QGroupBox::title {{
            color: {COLORS['accent_blue']};
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 8px;
        }}
        
        QPushButton {{
            background-color: {COLORS['bg_tertiary']};
            border: 1px solid {COLORS['border']};
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: 500;
            min-height: 32px;
        }}
        
        QPushButton:hover {{
            background-color: {COLORS['border']};
            border-color: {COLORS['accent_blue']};
        }}
        
        QPushButton:disabled {{
            background-color: {COLORS['bg_secondary']};
            color: {COLORS['text_secondary']};
        }}
        
        QTextEdit, QTextBrowser {{
            background-color: {COLORS['bg_secondary']};
            border: 1px solid {COLORS['border']};
            border-radius: 6px;
            padding: 8px;
        }}
        
        QLineEdit {{
            background-color: {COLORS['bg_secondary']};
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
            padding: 6px;
        }}
        
        QComboBox {{
            background-color: {COLORS['bg_secondary']};
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
            padding: 6px;
            min-width: 120px;
        }}
        
        QComboBox::drop-down {{
            border: none;
        }}
        
        QComboBox::down-arrow {{
            image: none;
            width: 0;
            height: 0;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 6px solid {COLORS['text_secondary']};
            margin-right: 8px;
        }}
        
        QProgressBar {{
            background-color: {COLORS['bg_tertiary']};
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
            text-align: center;
        }}
        
        QProgressBar::chunk {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {COLORS['accent_blue']}, stop:1 {COLORS['accent_cyan']});
            border-radius: 3px;
        }}
        
        QTabWidget::pane {{
            background-color: {COLORS['bg_secondary']};
            border: 1px solid {COLORS['border']};
            border-radius: 6px;
        }}
        
        QTabBar::tab {{
            background-color: {COLORS['bg_tertiary']};
            padding: 8px 16px;
            margin-right: 4px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            border-bottom: 2px solid transparent;
        }}
        
        QTabBar::tab:selected {{
            background-color: {COLORS['bg_secondary']};
            border-bottom: 2px solid {COLORS['accent_blue']};
        }}
        
        QListWidget {{
            background-color: {COLORS['bg_secondary']};
            border: 1px solid {COLORS['border']};
            border-radius: 6px;
        }}
        
        QCheckBox {{
            color: {COLORS['text_primary']};
        }}
        
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border: 1px solid {COLORS['border']};
            border-radius: 3px;
            background-color: {COLORS['bg_tertiary']};
        }}
        
        QCheckBox::indicator:checked {{
            background-color: {COLORS['accent_blue']};
            border-color: {COLORS['accent_blue']};
        }}
        
        QSlider::groove:horizontal {{
            background: {COLORS['bg_tertiary']};
            height: 6px;
            border-radius: 3px;
        }}
        
        QSlider::handle:horizontal {{
            background: {COLORS['accent_blue']};
            width: 16px;
            height: 16px;
            border-radius: 8px;
            margin: -5px 0;
        }}
        
        QStatusBar {{
            background-color: {COLORS['bg_secondary']};
            color: {COLORS['text_secondary']};
        }}
        """

# =====================================
# MAIN ENTRY POINT
# =====================================

def main():
    """Main entry point for Aurora Complete System"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application metadata
    app.setApplicationName("Aurora Complete System")
    app.setOrganizationName("Aurora AI")
    
    # Create and show main window
    window = AuroraCompleteWindow()
    window.show()
    
    # Start event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    banner_path = AURORA_STORAGE_PATH
    # Truncate path for display if it's too long
    if len(banner_path) > 60:
        banner_path = "..." + banner_path[-57:]
        
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║         AURORA COMPLETE SYSTEM - ENHANCED MEMORY & LEARNING         ║
║               With Ollama, SQL Storage & Self-Development           ║
║                     FULL PRODUCTION VERSION 2.3                     ║
║                                                                      ║
║  Features:                                                           ║
║  • Persistent SQL-based memory storage                              ║
║  • Semantic search with embeddings                                  ║
║  • Pattern detection and learning                                   ║
║  • Knowledge graph construction                                     ║
║  • Self-development and meta-reflection                            ║
║  • 7-dimensional processing                                         ║
║  • Full Ollama integration                                          ║
║                                                                      ║
║  Storage: {banner_path.ljust(68)}║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    main()

