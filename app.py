"""
Comprehensive Dynamic Knowledge Graph System - FINAL WORKING VERSION
Integrates: Document Processing, Image OCR, Incremental Updates, Multi-Language Translation (GPT-4.1 Only), GraphRAG
"""

import streamlit as st
import json
import hashlib
import io
import base64
import re
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum

# Azure imports
from azure.storage.blob import BlobServiceClient
from azure.cosmos import CosmosClient
from gremlin_python.driver import client, serializer
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient

# Computer Vision for OCR
try:
    from azure.cognitiveservices.vision.computervision import ComputerVisionClient
    from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
    from msrest.authentication import CognitiveServicesCredentials
    COMPUTER_VISION_AVAILABLE = True
except ImportError:
    COMPUTER_VISION_AVAILABLE = False
    st.warning("Computer Vision not available. Install: pip install azure-cognitiveservices-vision-computervision")

# Document processing
try:
    import docx
    from pypdf import PdfReader
    DOCUMENT_PROCESSING_AVAILABLE = True
except ImportError:
    DOCUMENT_PROCESSING_AVAILABLE = False
    st.warning("Document processing not available. Install: pip install python-docx pypdf")

# FIXED Configuration Class - All Working Services
class Config:
    # Storage Account
    STORAGE_ACCOUNT_NAME = "knowledgeold"
    STORAGE_ACCOUNT_KEY = "QCOR3ruMelY0iLB8AoIsX2plfDHhKCbejzoRljVS58HNN7zROCsq1aRDHrCV45aCEOophy4NhH7U+AStOasWrA=="
    CONTAINER_NAME = "knowledgebase-old"
    
    # Cosmos DB SQL (for document tracking)
    COSMOS_SQL_ENDPOINT = "https://idefixapp-dev.documents.azure.com:443/"
    COSMOS_SQL_KEY = "iUy3qNUqFN5xoMTQnpK1NQ3SAuFuh0hcvHzvKLv98pnOMPwqhKipiaudPoiJeBjqlTpE9vxH6WpXACDb2CO1LA=="
    
    # Cosmos DB Gremlin
    GREMLIN_ENDPOINT = "wss://idefixapp-dev-graph.gremlin.cosmos.azure.com:443/"
    GREMLIN_KEY = "gLEHBNYhwj15qcnz2uwNbU2DahEen9vX48EGuXZkpWIWG2xOWB3HCa9q966BGhU5Ilt05qJo9cyOACDbAQsPbA=="
    GREMLIN_DATABASE = "GraphDB"
    GREMLIN_COLLECTION = "GraphDB-id"
    
    # FIXED: Azure OpenAI GPT-4.1 Configuration 
    OPENAI_ENDPOINT = "https://azure-ai-foundry-genai.cognitiveservices.azure.com/"
    OPENAI_KEY = "4VLn9irDByzTBaiFVFVKwp4dO1yZQRetUIhAEgryX1hSARocLnqwJQQJ99BEACHYHv6XJ3w3AAAAACOGegEq"
    OPENAI_DEPLOYMENT = "gpt-4.1"  # FIXED: Use your actual working deployment
    OPENAI_API_VERSION = "2024-12-01-preview"
    
    # Azure AI Search
    SEARCH_ENDPOINT = "https://azure-aisearch-idefixai.search.windows.net"
    SEARCH_KEY = "ghmaljxX1Wo8yLc79YaIhce6E0EH5NzmahgL1A0xlyAzSeBIXhds"
    SEARCH_INDEX_NAME = "knowledge-base-index"
    
    # FIXED: Computer Vision with your actual credentials
    COMPUTER_VISION_ENDPOINT = "https://computervisionservice-idefixai.cognitiveservices.azure.com/"
    COMPUTER_VISION_KEY = "BvRnrh4lnIKGIsWM2MUhsfRBKWsODSEwvrefXmm5SczUio5nyzk2JQQJ99BGACHYHv6XJ3w3AAAFACOGBwxp"

# Language Support
class SupportedLanguage(Enum):
    ENGLISH = "English"
    ARABIC = "Arabic" 
    CHINESE = "Chinese (Simplified)"
    JAPANESE = "Japanese"

# Utility Functions
def safe_json_dumps(obj, max_depth=5, current_depth=0):
    """Safely convert object to JSON string with circular reference protection"""
    if current_depth > max_depth:
        return "<Max depth reached>"
    
    try:
        if isinstance(obj, dict):
            safe_dict = {}
            for key, value in obj.items():
                try:
                    safe_dict[str(key)] = safe_json_dumps(value, max_depth, current_depth + 1)
                except Exception:
                    safe_dict[str(key)] = "<Error serializing value>"
            return safe_dict
        elif isinstance(obj, list):
            safe_list = []
            for item in obj[:10]:
                try:
                    safe_list.append(safe_json_dumps(item, max_depth, current_depth + 1))
                except Exception:
                    safe_list.append("<Error serializing item>")
            if len(obj) > 10:
                safe_list.append(f"<... and {len(obj) - 10} more items>")
            return safe_list
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)
    except Exception as e:
        return f"<Error: {str(e)}>"

# Document Processing Classes
class DocumentProcessor:
    """Extract text from various document formats including images"""
    
    def __init__(self, computer_vision_client=None):
        self.computer_vision_client = computer_vision_client
    
    def extract_text_from_docx(self, blob_content: bytes) -> str:
        """Extract text from DOCX files"""
        if not DOCUMENT_PROCESSING_AVAILABLE:
            return "Document processing not available"
        try:
            doc = docx.Document(io.BytesIO(blob_content))
            text = []
            for paragraph in doc.paragraphs:
                text.append(paragraph.text)
            return '\n'.join(text)
        except Exception as e:
            return f"Error extracting DOCX: {e}"
    
    def extract_text_from_pdf(self, blob_content: bytes) -> str:
        """Extract text from PDF files"""
        if not DOCUMENT_PROCESSING_AVAILABLE:
            return "Document processing not available"
        try:
            pdf_reader = PdfReader(io.BytesIO(blob_content))
            text = []
            for page in pdf_reader.pages:
                text.append(page.extract_text())
            return '\n'.join(text)
        except Exception as e:
            return f"Error extracting PDF: {e}"
    
    def extract_text_from_txt(self, blob_content: bytes) -> str:
        """Extract text from TXT files"""
        try:
            return blob_content.decode('utf-8')
        except Exception as e:
            return f"Error extracting TXT: {e}"
    
    def extract_text_from_image(self, blob_content: bytes, image_name: str) -> Dict[str, Any]:
        """Extract text from images using OCR"""
        if not self.computer_vision_client:
            return {
                "extracted_text": "",
                "description": "Computer Vision not configured",
                "tags": [],
                "objects": [],
                "confidence": 0,
                "image_name": image_name
            }
        
        try:
            image_stream = io.BytesIO(blob_content)
            
            # OCR
            ocr_result = self.computer_vision_client.read_in_stream(image_stream, raw=True)
            operation_id = ocr_result.headers["Operation-Location"].split("/")[-1]
            
            # Wait for results
            while True:
                result = self.computer_vision_client.get_read_result(operation_id)
                if result.status not in ['notStarted', 'running']:
                    break
                time.sleep(1)
            
            # Extract text
            extracted_text = ""
            if result.status == OperationStatusCodes.succeeded:
                for text_result in result.analyze_result.read_results:
                    for line in text_result.lines:
                        extracted_text += line.text + "\n"
            
            # Image analysis
            image_analysis = self.computer_vision_client.analyze_image_in_stream(
                io.BytesIO(blob_content),
                visual_features=['Description', 'Tags', 'Objects']
            )
            
            return {
                "extracted_text": extracted_text,
                "description": image_analysis.description.captions[0].text if image_analysis.description.captions else "",
                "tags": [tag.name for tag in image_analysis.tags],
                "objects": [obj.object_property for obj in image_analysis.objects] if image_analysis.objects else [],
                "confidence": image_analysis.description.captions[0].confidence if image_analysis.description.captions else 0,
                "image_name": image_name
            }
            
        except Exception as e:
            return {
                "extracted_text": "",
                "description": f"Error processing image: {str(e)}",
                "tags": [],
                "objects": [],
                "confidence": 0,
                "image_name": image_name
            }
    
    def process_blob(self, blob_name: str, blob_content: bytes) -> Dict[str, Any]:
        """Process any type of blob - text document or image"""
        filename_lower = blob_name.lower()
        
        # Text documents
        if filename_lower.endswith(('.docx', '.pdf', '.txt', '.md')):
            return self._process_text_document(blob_name, blob_content)
        # Images
        elif filename_lower.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            return self._process_image_document(blob_name, blob_content)
        else:
            return {
                "text": f"Unsupported file type: {blob_name}",
                "type": "unsupported"
            }
    
    def _process_text_document(self, blob_name: str, blob_content: bytes) -> Dict[str, Any]:
        """Process text documents"""
        filename_lower = blob_name.lower()
        
        if filename_lower.endswith('.docx'):
            text = self.extract_text_from_docx(blob_content)
        elif filename_lower.endswith('.pdf'):
            text = self.extract_text_from_pdf(blob_content)
        elif filename_lower.endswith(('.txt', '.md')):
            text = self.extract_text_from_txt(blob_content)
        else:
            text = ""
        
        return {"text": text, "type": "text_document"}
    
    def _process_image_document(self, blob_name: str, blob_content: bytes) -> Dict[str, Any]:
        """Process image documents using OCR"""
        image_result = self.extract_text_from_image(blob_content, blob_name)
        
        combined_text = f"""
        Image: {blob_name}
        
        Extracted Text:
        {image_result['extracted_text']}
        
        Image Description: {image_result['description']}
        
        Tags: {', '.join(image_result['tags'])}
        
        Objects Detected: {', '.join(image_result['objects'])}
        """
        
        return {
            "text": combined_text,
            "type": "image_document",
            "image_metadata": image_result
        }

# Language Detection and Translation
class LanguageDetector:
    """Detect language of input text"""
    
    @staticmethod
    def detect_language(text: str) -> SupportedLanguage:
        """Simple language detection based on character patterns"""
        # Arabic detection
        arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
        if arabic_pattern.search(text):
            return SupportedLanguage.ARABIC
        
        # Chinese detection
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        if chinese_pattern.search(text):
            return SupportedLanguage.CHINESE
        
        # Japanese detection
        japanese_pattern = re.compile(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf]')
        if japanese_pattern.search(text):
            return SupportedLanguage.JAPANESE
        
        return SupportedLanguage.ENGLISH

# FIXED: GPT-4.1 Only Translation Service
class GPT41OnlyTranslationService:
    """Translation service using ONLY GPT-4.1 for all languages (no JAIS)"""
    
    def __init__(self, openai_client: AzureOpenAI):
        self.openai_client = openai_client
        self.language_detector = LanguageDetector()
    
    def translate_text(self, text: str, target_language: SupportedLanguage, 
                      source_language: Optional[SupportedLanguage] = None) -> Dict[str, Any]:
        """Translate text using GPT-4.1 for all languages"""
        
        if not source_language:
            source_language = self.language_detector.detect_language(text)
        
        if source_language == target_language:
            return {
                "translated_text": text,
                "source_language": source_language.value,
                "target_language": target_language.value,
                "model_used": "none",
                "confidence": 1.0
            }
        
        try:
            return self._translate_with_gpt41(text, source_language, target_language)
        except Exception as e:
            return {
                "translated_text": text,
                "source_language": source_language.value,
                "target_language": target_language.value,
                "model_used": "error",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _translate_with_gpt41(self, text: str, source_language: SupportedLanguage, 
                             target_language: SupportedLanguage) -> Dict[str, Any]:
        """Translate using GPT-4.1 for all languages including Arabic"""
        
        language_names = {
            SupportedLanguage.CHINESE: "Chinese (Simplified)",
            SupportedLanguage.JAPANESE: "Japanese",
            SupportedLanguage.ARABIC: "Arabic",
            SupportedLanguage.ENGLISH: "English"
        }
        
        target_lang_name = language_names.get(target_language, target_language.value)
        source_lang_name = language_names.get(source_language, source_language.value)
        
        # Special prompt for Arabic
        if target_language == SupportedLanguage.ARABIC:
            prompt = f"""
            Translate the following text from {source_lang_name} to Arabic with high accuracy:

            Original text:
            {text}

            Translation requirements:
            1. Use Modern Standard Arabic (فصحى)
            2. Maintain technical terminology appropriately  
            3. Preserve proper nouns and brand names
            4. Ensure natural, fluent Arabic
            5. Keep the original meaning and context

            Arabic translation:
            """
        else:
            prompt = f"""
            Translate the following text from {source_lang_name} to {target_lang_name} with high accuracy:

            Original text:
            {text}

            Translation requirements:
            1. Maintain the original meaning and context
            2. Use natural, fluent {target_lang_name}
            3. Preserve technical terms appropriately
            4. Keep proper nouns and brand names as appropriate
            5. Ensure cultural appropriateness

            {target_lang_name} translation:
            """
        
        try:
            response = self.openai_client.chat.completions.create(
                model=Config.OPENAI_DEPLOYMENT,  # Uses "gpt-4.1"
                messages=[
                    {"role": "system", "content": f"You are an expert translator specializing in accurate {target_lang_name} translations, particularly for technical and business content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            return {
                "translated_text": response.choices[0].message.content.strip(),
                "source_language": source_language.value,
                "target_language": target_language.value,
                "model_used": "gpt-4.1",
                "confidence": 0.9
            }
        except Exception as e:
            raise Exception(f"GPT-4.1 translation failed: {str(e)}")

# Document Tracking for Incremental Processing
class DocumentTracker:
    """Track processed documents to enable incremental processing"""
    
    def __init__(self, cosmos_client: CosmosClient, database_name: str = "GraphDB"):
        self.cosmos_client = cosmos_client
        self.database_name = database_name
        self.container_name = "document_tracking"
        self._ensure_tracking_container()
    
    def _ensure_tracking_container(self):
        """Create tracking container if it doesn't exist"""
        try:
            database = self.cosmos_client.get_database_client(self.database_name)
            try:
                database.get_container_client(self.container_name)
            except:
                database.create_container(
                    id=self.container_name,
                    partition_key={"paths": ["/document_name"], "kind": "Hash"}
                )
                print(f"✅ Created tracking container: {self.container_name}")
        except Exception as e:
            print(f"❌ Error setting up tracking container: {e}")
    
    def get_document_hash(self, blob_content: bytes) -> str:
        """Calculate hash of document content"""
        return hashlib.sha256(blob_content).hexdigest()
    
    def is_document_processed(self, document_name: str, content_hash: str) -> bool:
        """Check if document with this hash has been processed"""
        try:
            database = self.cosmos_client.get_database_client(self.database_name)
            container = database.get_container_client(self.container_name)
            
            query = f"SELECT * FROM c WHERE c.document_name = '{document_name}'"
            items = list(container.query_items(query, enable_cross_partition_query=True))
            
            if items:
                stored_hash = items[0].get('content_hash')
                return stored_hash == content_hash
            
            return False
        except Exception as e:
            print(f"❌ Error checking document status: {e}")
            return False
    
    def mark_document_processed(self, document_name: str, content_hash: str, 
                              entities_created: List[str], processing_info: Dict):
        """Mark document as processed with metadata"""
        try:
            database = self.cosmos_client.get_database_client(self.database_name)
            container = database.get_container_client(self.container_name)
            
            document_record = {
                "id": f"doc_{hashlib.md5(document_name.encode()).hexdigest()}",
                "document_name": document_name,
                "content_hash": content_hash,
                "processed_at": datetime.now().isoformat(),
                "entities_created": entities_created,
                "entity_count": len(entities_created),
                "processing_info": processing_info
            }
            
            container.upsert_item(document_record)
            print(f"✅ Marked {document_name} as processed")
        except Exception as e:
            print(f"❌ Error marking document as processed: {e}")

# AI Entity Extractor
class AIEntityExtractor:
    """Use AI to extract entities and relationships from documents"""
    
    def __init__(self, openai_client: AzureOpenAI):
        self.openai_client = openai_client
    
    def extract_entities_and_relationships(self, text: str, document_name: str) -> Dict[str, Any]:
        """Extract entities and relationships using AI"""
        max_chunk_size = 3000
        if len(text) > max_chunk_size:
            text = text[:max_chunk_size] + "..."
        
        prompt = f"""
        Analyze the following document and extract entities and relationships. Focus on:
        
        1. TECHNOLOGIES (software, hardware, platforms, frameworks)
        2. COMPANIES/ORGANIZATIONS 
        3. PEOPLE (experts, researchers, executives)
        4. CONCEPTS (methodologies, processes, strategies)
        5. PRODUCTS/SERVICES
        6. LOCATIONS (if relevant)
        
        Document: "{document_name}"
        Content: {text}
        
        Return a JSON response with this exact structure:
        {{
            "entities": [
                {{
                    "name": "Entity Name",
                    "type": "technology|company|person|concept|product|location",
                    "description": "Brief description",
                    "properties": {{
                        "category": "specific category",
                        "mentioned_context": "how it's mentioned in the document"
                    }}
                }}
            ],
            "relationships": [
                {{
                    "source": "Source Entity Name",
                    "target": "Target Entity Name", 
                    "relationship_type": "uses|implements|competes_with|part_of|developed_by|related_to",
                    "description": "Description of the relationship",
                    "properties": {{
                        "strength": "strong|medium|weak",
                        "context": "context from document"
                    }}
                }}
            ],
            "document_summary": "Brief summary of the document content"
        }}
        
        Be specific and accurate. Only extract entities that are clearly mentioned in the text.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model=Config.OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting structured information from technical documents. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            try:
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                json_content = content[json_start:json_end]
                
                extracted_data = json.loads(json_content)
                extracted_data['source_document'] = document_name
                extracted_data['extraction_timestamp'] = datetime.now().isoformat()
                
                return extracted_data
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                return self._create_fallback_extraction(text, document_name)
        except Exception as e:
            print(f"AI extraction error: {e}")
            return self._create_fallback_extraction(text, document_name)
    
    def _create_fallback_extraction(self, text: str, document_name: str) -> Dict[str, Any]:
        """Fallback extraction if AI fails"""
        tech_keywords = ['azure', 'python', 'ai', 'machine learning', 'cloud', 'database', 'api', 'quantum', 'nvidia']
        company_keywords = ['microsoft', 'google', 'amazon', 'openai', 'nvidia', 'apple']
        
        entities = []
        text_lower = text.lower()
        
        for keyword in tech_keywords:
            if keyword in text_lower:
                entities.append({
                    "name": keyword.title(),
                    "type": "technology",
                    "description": f"Technology mentioned in {document_name}",
                    "properties": {"category": "technology", "mentioned_context": "document content"}
                })
        
        for keyword in company_keywords:
            if keyword in text_lower:
                entities.append({
                    "name": keyword.title(),
                    "type": "company", 
                    "description": f"Company mentioned in {document_name}",
                    "properties": {"category": "organization", "mentioned_context": "document content"}
                })
        
        return {
            "entities": entities,
            "relationships": [],
            "document_summary": f"Fallback extraction from {document_name}",
            "source_document": document_name,
            "extraction_timestamp": datetime.now().isoformat()
        }

# Knowledge Graph Builder
class KnowledgeGraphBuilder:
    """Build knowledge graph in Cosmos DB Gremlin"""
    
    def __init__(self, gremlin_client):
        self.gremlin_client = gremlin_client
    
    def create_entity_vertex(self, entity: Dict[str, Any], document_name: str) -> str:
        """Create a vertex for an entity"""
        entity_id = self._create_entity_id(entity['name'], entity['type'])
        
        name = entity['name'].replace("'", "\\'")
        description = entity.get('description', '').replace("'", "\\'")
        entity_type = entity['type']
        
        gremlin_query = f"""
        g.V().has('id', '{entity_id}').fold().coalesce(
            unfold(),
            addV('{entity_type}')
                .property('id', '{entity_id}')
                .property('name', '{name}')
                .property('type', '{entity_type}')
                .property('description', '{description}')
                .property('source_document', '{document_name}')
                .property('created_at', '{datetime.now().isoformat()}')
        )
        """
        
        for key, value in entity.get('properties', {}).items():
            if isinstance(value, str):
                value = value.replace("'", "\\'")
                gremlin_query += f".property('{key}', '{value}')"
        
        try:
            result = self.gremlin_client.submit(gremlin_query)
            result.all().result()
            print(f"✅ Created/Updated entity: {name}")
            return entity_id
        except Exception as e:
            print(f"❌ Error creating entity {name}: {e}")
            return None
    
    def create_relationship_edge(self, relationship: Dict[str, Any], source_id: str, target_id: str):
        """Create an edge for a relationship"""
        rel_type = relationship['relationship_type']
        description = relationship.get('description', '').replace("'", "\\'")
        
        gremlin_query = f"""
        g.V().has('id', '{source_id}').as('source')
         .V().has('id', '{target_id}').as('target')
         .addE('{rel_type}')
         .from('source').to('target')
         .property('description', '{description}')
         .property('created_at', '{datetime.now().isoformat()}')
        """
        
        for key, value in relationship.get('properties', {}).items():
            if isinstance(value, str):
                value = value.replace("'", "\\'")
                gremlin_query += f".property('{key}', '{value}')"
        
        try:
            result = self.gremlin_client.submit(gremlin_query)
            result.all().result()
            print(f"✅ Created relationship: {rel_type}")
        except Exception as e:
            print(f"❌ Error creating relationship: {e}")
    
    def create_document_vertex(self, document_name: str, summary: str, entities: List[str]):
        """Create a vertex for the document itself"""
        doc_id = f"doc_{hashlib.md5(document_name.encode()).hexdigest()}"
        summary_clean = summary.replace("'", "\\'")
        
        gremlin_query = f"""
        g.V().has('id', '{doc_id}').fold().coalesce(
            unfold(),
            addV('document')
                .property('id', '{doc_id}')
                .property('name', '{document_name}')
                .property('type', 'document')
                .property('summary', '{summary_clean}')
                .property('entity_count', {len(entities)})
                .property('processed_at', '{datetime.now().isoformat()}')
        )
        """
        
        try:
            result = self.gremlin_client.submit(gremlin_query)
            result.all().result()
            print(f"✅ Created document vertex: {document_name}")
            
            for entity_id in entities:
                if entity_id:
                    link_query = f"""
                    g.V().has('id', '{doc_id}').as('doc')
                     .V().has('id', '{entity_id}').as('entity')
                     .addE('mentions')
                     .from('doc').to('entity')
                     .property('created_at', '{datetime.now().isoformat()}')
                    """
                    try:
                        self.gremlin_client.submit(link_query).all().result()
                    except Exception as e:
                        print(f"Warning: Could not link document to entity {entity_id}: {e}")
        except Exception as e:
            print(f"❌ Error creating document vertex: {e}")
    
    def _create_entity_id(self, name: str, entity_type: str) -> str:
        """Create a unique ID for an entity"""
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', name.lower())
        return f"{entity_type}_{clean_name}"

# GraphRAG Query System
class AzureCosmosGraphQuerier:
    """Graph querier specifically designed for Azure Cosmos DB Gremlin API"""
    
    def __init__(self, gremlin_client, openai_client: AzureOpenAI):
        self.gremlin_client = gremlin_client
        self.openai_client = openai_client
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Simple query analysis without external AI calls"""
        query_lower = query.lower()
        
        entities = []
        tech_terms = ['python', 'azure', 'ai', 'nvidia', 'quantum', 'cloud', 'machine learning', 'deep learning']
        
        words = query.split()
        for word in words:
            clean_word = re.sub(r'[^a-zA-Z]', '', word)
            if len(clean_word) > 2:
                if clean_word[0].isupper() or clean_word.lower() in tech_terms:
                    entities.append(clean_word)
        
        if any(word in query_lower for word in ['how', 'relationship', 'related', 'connect']):
            intent = "find_relationships"
        elif any(word in query_lower for word in ['what', 'tell me about', 'describe']):
            intent = "search_entity"
        elif any(word in query_lower for word in ['compare', 'difference', 'versus']):
            intent = "compare_entities"
        else:
            intent = "general_question"
        
        return {
            "intent": intent,
            "entities": entities,
            "topics": entities,
            "relationship_types": ["uses", "implements", "related_to"],
            "entity_types": ["technology", "company", "concept"],
            "search_strategy": "exact_match"
        }
    
    def search_knowledge_graph(self, query: str) -> Dict[str, Any]:
        """Search using Azure Cosmos DB compatible queries"""
        try:
            print(f"🔍 Processing query: {query}")
            
            analysis = self.analyze_query_intent(query)
            
            print(f"🎯 Detected entities: {analysis['entities']}")
            print(f"📊 Intent: {analysis['intent']}")
            
            results = {
                "query": query,
                "analysis": analysis,
                "found_entities": [],
                "relationships": [],
                "documents": [],
                "graph_data": None,
                "error": None
            }
            
            if analysis['entities']:
                entity_results = self._search_entities_azure_compatible(analysis['entities'])
                results['found_entities'] = entity_results
                
                if entity_results:
                    relationship_results = self._get_relationships_azure_compatible(entity_results)
                    results['relationships'] = relationship_results
                    
                    document_results = self._get_documents_azure_compatible(entity_results)
                    results['documents'] = document_results
            
            if not results['found_entities']:
                general_results = self._general_exploration_azure_compatible()
                results['found_entities'] = general_results
            
            results['graph_data'] = self._format_graph_response(results)
            return results
            
        except Exception as e:
            print(f"Search error: {e}")
            return {
                "query": query,
                "analysis": {"intent": "error", "entities": [], "topics": []},
                "found_entities": [],
                "relationships": [],
                "documents": [],
                "graph_data": {"summary": f"Search failed: {str(e)}", "error": str(e)},
                "error": str(e)
            }
    
    def _search_entities_azure_compatible(self, entities: List[str]) -> List[Dict[str, Any]]:
        """Search entities using only Azure Cosmos DB supported functions"""
        found_entities = []
        
        for entity in entities:
            print(f"🔍 Searching for entity: {entity}")
            
            search_queries = [
                f"g.V().has('name', '{entity}').limit(3)",
                f"g.V().has('name', '{entity.lower()}').limit(3)",
                f"g.V().has('name', '{entity.upper()}').limit(3)",
                f"g.V().has('name', '{entity.title()}').limit(3)",
            ]
            
            for query_template in search_queries:
                try:
                    print(f"   Trying query: {query_template}")
                    result = self.gremlin_client.submit(query_template + ".valueMap()")
                    entities_found = result.all().result()
                    
                    if entities_found:
                        print(f"   ✅ Found {len(entities_found)} results")
                        for entity_data in entities_found:
                            safe_entity_data = safe_json_dumps(entity_data)
                            found_entities.append({
                                'search_term': entity,
                                'entity_data': safe_entity_data,
                                'query_used': query_template
                            })
                        break
                except Exception as e:
                    print(f"   ❌ Query failed: {e}")
                    continue
            
            # Broad search if exact match fails
            if not any(e['search_term'] == entity for e in found_entities):
                try:
                    broad_query = "g.V().limit(20).valueMap()"
                    print(f"   Trying broad search: {broad_query}")
                    result = self.gremlin_client.submit(broad_query)
                    all_entities = result.all().result()
                    
                    for entity_data in all_entities:
                        try:
                            name_field = entity_data.get('name', [])
                            if isinstance(name_field, list) and len(name_field) > 0:
                                name = str(name_field[0]).lower()
                                if entity.lower() in name or name in entity.lower():
                                    print(f"   ✅ Found broad match: {name}")
                                    safe_entity_data = safe_json_dumps(entity_data)
                                    found_entities.append({
                                        'search_term': entity,
                                        'entity_data': safe_entity_data,
                                        'query_used': 'broad_search'
                                    })
                                    break
                        except Exception as e:
                            print(f"   Warning: Error checking entity {entity_data}: {e}")
                            continue
                except Exception as e:
                    print(f"   ❌ Broad search failed: {e}")
        
        print(f"🎯 Total entities found: {len(found_entities)}")
        return found_entities
    
    def _get_relationships_azure_compatible(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get relationships using Azure-compatible queries"""
        relationships = []
        
        for entity in entities:
            try:
                entity_data = entity.get('entity_data', {})
                entity_id = None
                
                if isinstance(entity_data, dict):
                    if 'id' in entity_data:
                        id_value = entity_data['id']
                        if isinstance(id_value, list) and len(id_value) > 0:
                            entity_id = id_value[0]
                        elif isinstance(id_value, str):
                            entity_id = id_value
                
                if not entity_id:
                    print(f"⚠️  Could not extract entity ID from: {entity_data}")
                    continue
                
                print(f"🔗 Getting relationships for: {entity_id}")
                
                # Get outgoing edges
                try:
                    outgoing_query = f"g.V().has('id', '{entity_id}').outE().limit(5)"
                    print(f"   Outgoing query: {outgoing_query}")
                    
                    result = self.gremlin_client.submit(outgoing_query + ".project('label', 'inV_name').by(label()).by(inV().values('name').fold())")
                    outgoing_rels = result.all().result()
                    
                    for rel in outgoing_rels:
                        safe_rel = safe_json_dumps(rel)
                        relationships.append({
                            'source_entity': entity_id,
                            'relationship_data': safe_rel,
                            'direction': 'outgoing'
                        })
                        print(f"   ✅ Found outgoing relationship: {rel}")
                except Exception as e:
                    print(f"   ❌ Outgoing relationships error: {e}")
                
                # Get incoming edges
                try:
                    incoming_query = f"g.V().has('id', '{entity_id}').inE().limit(5)"
                    print(f"   Incoming query: {incoming_query}")
                    
                    result = self.gremlin_client.submit(incoming_query + ".project('label', 'outV_name').by(label()).by(outV().values('name').fold())")
                    incoming_rels = result.all().result()
                    
                    for rel in incoming_rels:
                        safe_rel = safe_json_dumps(rel)
                        relationships.append({
                            'source_entity': entity_id,
                            'relationship_data': safe_rel,
                            'direction': 'incoming'
                        })
                        print(f"   ✅ Found incoming relationship: {rel}")
                except Exception as e:
                    print(f"   ❌ Incoming relationships error: {e}")
                    
            except Exception as e:
                print(f"❌ Entity relationship error: {e}")
                continue
        
        print(f"🔗 Total relationships found: {len(relationships)}")
        return relationships
    
    def _get_documents_azure_compatible(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get related documents using Azure-compatible queries"""
        documents = []
        
        for entity in entities:
            try:
                entity_data = entity.get('entity_data', {})
                entity_id = None
                
                if isinstance(entity_data, dict):
                    if 'id' in entity_data:
                        id_value = entity_data['id']
                        if isinstance(id_value, list) and len(id_value) > 0:
                            entity_id = id_value[0]
                        elif isinstance(id_value, str):
                            entity_id = id_value
                
                if not entity_id:
                    continue
                
                doc_query = f"g.V().has('id', '{entity_id}').inE('mentions').outV().hasLabel('document').limit(3).valueMap()"
                print(f"📄 Document query: {doc_query}")
                
                result = self.gremlin_client.submit(doc_query)
                docs = result.all().result()
                
                for doc in docs:
                    safe_doc = safe_json_dumps(doc)
                    documents.append({
                        'entity_id': entity_id,
                        'document_data': safe_doc
                    })
                    print(f"   ✅ Found document: {doc.get('name', ['Unknown'])}")
            except Exception as e:
                print(f"📄 Document search error: {e}")
                continue
        
        print(f"📄 Total documents found: {len(documents)}")
        return documents
    
    def _general_exploration_azure_compatible(self) -> List[Dict[str, Any]]:
        """General exploration using Azure-compatible queries"""
        try:
            print("🔍 Performing general exploration...")
            
            exploration_queries = [
                ("g.V().hasLabel('technology').limit(3).valueMap()", "technology"),
                ("g.V().hasLabel('company').limit(3).valueMap()", "company"),
                ("g.V().hasLabel('concept').limit(3).valueMap()", "concept"),
                ("g.V().limit(5).valueMap()", "general")
            ]
            
            general_results = []
            
            for query, entity_type in exploration_queries:
                try:
                    print(f"   Trying {entity_type} query: {query}")
                    result = self.gremlin_client.submit(query)
                    entities = result.all().result()
                    
                    for entity in entities:
                        safe_entity = safe_json_dumps(entity)
                        general_results.append({
                            'search_term': f'exploration_{entity_type}',
                            'entity_data': safe_entity,
                            'query_used': query
                        })
                        print(f"   ✅ Found {entity_type}: {entity.get('name', ['Unknown'])}")
                except Exception as e:
                    print(f"   ❌ {entity_type} query failed: {e}")
                    continue
            
            return general_results
        except Exception as e:
            print(f"❌ General exploration error: {e}")
            return []
    
    def _format_graph_response(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Format the graph search results for consumption"""
        try:
            formatted = {
                "summary": self._create_summary(results),
                "entities_found": len(results['found_entities']),
                "relationships_found": len(results['relationships']),
                "documents_found": len(results['documents']),
                "key_findings": self._extract_key_findings(results),
                "error": results.get('error'),
                "raw_data": safe_json_dumps(results, max_depth=2)
            }
            return formatted
        except Exception as e:
            return {
                "summary": f"Error formatting results: {str(e)}",
                "entities_found": 0,
                "relationships_found": 0,
                "documents_found": 0,
                "key_findings": [],
                "error": str(e),
                "raw_data": {}
            }
    
    def _create_summary(self, results: Dict[str, Any]) -> str:
        """Create a summary of findings"""
        try:
            entity_count = len(results['found_entities'])
            rel_count = len(results['relationships'])
            doc_count = len(results['documents'])
            
            if results.get('error'):
                return f"Search encountered an error: {results['error']}"
            
            if entity_count == 0:
                return "No specific entities found in the knowledge graph for this query."
            
            summary = f"Found {entity_count} relevant entities"
            
            if rel_count > 0:
                summary += f" with {rel_count} relationships"
            
            if doc_count > 0:
                summary += f" mentioned in {doc_count} documents"
            
            return summary + "."
        except Exception as e:
            return f"Error creating summary: {str(e)}"
    
    def _extract_key_findings(self, results: Dict[str, Any]) -> List[str]:
        """Extract key findings from the results"""
        findings = []
        
        try:
            entities = results['found_entities'][:3]
            for entity in entities:
                try:
                    entity_data = entity.get('entity_data', {})
                    if isinstance(entity_data, dict):
                        name = entity_data.get('name', ['Unknown'])
                        if isinstance(name, list) and len(name) > 0:
                            name = name[0]
                        entity_type = entity_data.get('type', ['entity'])
                        if isinstance(entity_type, list) and len(entity_type) > 0:
                            entity_type = entity_type[0]
                        findings.append(f"Found {entity_type}: {name}")
                except Exception:
                    findings.append("Found entity (parsing error)")
            
            relationships = results['relationships'][:3]
            for rel in relationships:
                try:
                    rel_data = rel.get('relationship_data', {})
                    if isinstance(rel_data, dict):
                        rel_type = rel_data.get('label', 'related_to')
                        findings.append(f"Relationship: {rel_type}")
                except Exception:
                    findings.append("Found relationship")
        except Exception as e:
            findings.append(f"Error extracting findings: {str(e)}")
        
        return findings

# External Search Fallback
class ExternalSearchFallback:
    """External search when knowledge graph doesn't have information"""
    
    def __init__(self, openai_client: AzureOpenAI):
        self.openai_client = openai_client
    
    def search_external(self, query: str, graph_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Search external sources when graph search is insufficient"""
        context_info = ""
        if graph_context and graph_context.get('entities_found', 0) > 0:
            try:
                key_findings = graph_context.get('key_findings', [])
                context_info = f"""
                Note: The knowledge graph contains some related information:
                {json.dumps(key_findings, indent=2)}
                
                Please provide additional context and external information beyond what's in the knowledge graph.
                """
            except Exception:
                context_info = "Note: Some related information was found in the knowledge graph."
        
        prompt = f"""
        The user asked: "{query}"
        
        {context_info}
        
        Please provide a comprehensive answer using your knowledge and training data. 
        Focus on:
        1. Technical accuracy
        2. Current industry trends
        3. Practical applications
        4. Best practices
        5. Future outlook
        
        Be specific and provide actionable insights.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model=Config.OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "You are an expert technology and business consultant with access to comprehensive knowledge about technologies, companies, and industry trends."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            return {
                "source": "External Search",
                "content": response.choices[0].message.content,
                "context_used": bool(context_info),
                "query": query
            }
        except Exception as e:
            return {
                "source": "External Search",
                "content": f"External search failed: {str(e)}",
                "context_used": False,
                "query": query
            }

# Main Hybrid System
class HybridGraphRAGSystem:
    """Complete hybrid system with Azure Cosmos DB compatibility"""
    
    def __init__(self, gremlin_client, openai_client: AzureOpenAI):
        self.graph_querier = AzureCosmosGraphQuerier(gremlin_client, openai_client)
        self.external_search = ExternalSearchFallback(openai_client)
        self.openai_client = openai_client
    
    def query(self, user_query: str) -> Dict[str, Any]:
        """Main query method that combines graph and external search"""
        print(f"🚀 Processing query: {user_query}")
        
        try:
            graph_results = self.graph_querier.search_knowledge_graph(user_query)
            is_sufficient = self._evaluate_graph_results(graph_results)
            
            external_results = None
            if not is_sufficient:
                print("🌐 Graph results insufficient, searching external sources...")
                external_results = self.external_search.search_external(
                    user_query, 
                    graph_results.get('graph_data')
                )
            
            final_response = self._generate_hybrid_response(
                user_query, 
                graph_results, 
                external_results
            )
            
            return {
                "query": user_query,
                "graph_results": graph_results,
                "external_results": external_results,
                "final_response": final_response,
                "sources_used": self._get_sources_used(graph_results, external_results),
                "error": graph_results.get('error')
            }
        except Exception as e:
            print(f"❌ Query processing error: {e}")
            return {
                "query": user_query,
                "graph_results": {"error": str(e)},
                "external_results": None,
                "final_response": f"I encountered an error while processing your query: {str(e)}. Please try a simpler query or check the system status.",
                "sources_used": [],
                "error": str(e)
            }
    
    def _evaluate_graph_results(self, graph_results: Dict[str, Any]) -> bool:
        """Evaluate if graph results are sufficient to answer the query"""
        try:
            if graph_results.get('error'):
                return False
            
            entities_found = len(graph_results.get('found_entities', []))
            relationships_found = len(graph_results.get('relationships', []))
            
            print(f"📊 Evaluation: {entities_found} entities, {relationships_found} relationships")
            
            if entities_found >= 1 and relationships_found >= 1:
                print("✅ Graph results sufficient")
                return True
            
            if entities_found >= 2:
                print("✅ Graph results sufficient (multiple entities)")
                return True
            
            print("❌ Graph results insufficient")
            return False
        except Exception:
            return False
    
    def _generate_hybrid_response(self, query: str, graph_results: Dict[str, Any], external_results: Dict[str, Any] = None) -> str:
        """Generate final response combining graph and external results"""
        try:
            graph_data = safe_json_dumps(graph_results.get('graph_data', {}), max_depth=2)
            external_content = external_results.get('content', '') if external_results else ''
            
            prompt = f"""
            Generate a comprehensive response to this query: "{query}"
            
            Knowledge Graph Data:
            {json.dumps(graph_data, indent=2)}
            
            External Information:
            {external_content}
            
            Instructions:
            1. Start with information from the knowledge graph if available
            2. Supplement with external information to provide complete context
            3. Clearly cite sources: [Source: Knowledge Graph] and [Source: External Search]
            4. Structure the response logically
            5. Provide actionable insights and recommendations
            6. If the knowledge graph has limited information, acknowledge it and focus on external sources
            
            Be comprehensive but concise. Focus on answering the user's specific question.
            """
            
            response = self.openai_client.chat.completions.create(
                model=Config.OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "You are an expert assistant that synthesizes information from multiple sources to provide comprehensive, accurate answers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            try:
                graph_summary = graph_results.get('graph_data', {}).get('summary', 'No graph data found')
                return f"""
                Based on the available information:
                
                **From Knowledge Graph:** {graph_summary}
                
                **External Information:** {external_content if external_content else 'External search not available'}
                
                [Note: There was an error in response generation, but I've provided the basic information available]
                """
            except Exception:
                return f"I apologize, but I encountered an error while processing your query about '{query}'. Please try rephrasing your question or check the system status."
    
    def _get_sources_used(self, graph_results: Dict[str, Any], external_results: Dict[str, Any] = None) -> List[str]:
        """Get list of sources used in the response"""
        sources = []
        
        try:
            if graph_results.get('found_entities') and not graph_results.get('error'):
                sources.append("Knowledge Graph")
            
            if external_results:
                sources.append("External Search")
        except Exception:
            pass
        
        return sources

# Incremental Document Processor
class IncrementalDocumentProcessor:
    """Process only new/changed documents"""
    
    def __init__(self):
        # Initialize Azure clients
        self.blob_client = BlobServiceClient(
            account_url=f"https://{Config.STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
            credential=Config.STORAGE_ACCOUNT_KEY
        )
        
        self.cosmos_client = CosmosClient(
            Config.COSMOS_SQL_ENDPOINT,
            credential=Config.COSMOS_SQL_KEY
        )
        
        self.openai_client = AzureOpenAI(
            api_version=Config.OPENAI_API_VERSION,
            azure_endpoint=Config.OPENAI_ENDPOINT,
            api_key=Config.OPENAI_KEY
        )
        
        self.gremlin_client = client.Client(
            Config.GREMLIN_ENDPOINT,
            'g',
            username=f"/dbs/{Config.GREMLIN_DATABASE}/colls/{Config.GREMLIN_COLLECTION}",
            password=Config.GREMLIN_KEY,
            message_serializer=serializer.GraphSONSerializersV2d0()
        )
        
        # Initialize Computer Vision if available
        computer_vision_client = None
        if COMPUTER_VISION_AVAILABLE:
            try:
                computer_vision_client = ComputerVisionClient(
                    Config.COMPUTER_VISION_ENDPOINT,
                    CognitiveServicesCredentials(Config.COMPUTER_VISION_KEY)
                )
            except Exception as e:
                print(f"⚠️  Computer Vision not configured properly: {e}")
        
        # Initialize processors
        self.doc_tracker = DocumentTracker(self.cosmos_client)
        self.ai_extractor = AIEntityExtractor(self.openai_client)
        self.graph_builder = KnowledgeGraphBuilder(self.gremlin_client)
        self.doc_processor = DocumentProcessor(computer_vision_client)
    
    def process_new_documents(self, container_name: str = None) -> Dict[str, any]:
        """Process only new or changed documents"""
        if not container_name:
            container_name = Config.CONTAINER_NAME
        
        print("🔄 INCREMENTAL DOCUMENT PROCESSING")
        print("=" * 50)
        
        try:
            container_client = self.blob_client.get_container_client(container_name)
            blobs = list(container_client.list_blobs())
            
            results = {
                "total_blobs": len(blobs),
                "processed_count": 0,
                "skipped_count": 0,
                "error_count": 0,
                "new_documents": [],
                "skipped_documents": [],
                "error_documents": []
            }
            
            for blob in blobs:
                print(f"\n📄 Checking: {blob.name}")
                
                try:
                    blob_client = container_client.get_blob_client(blob.name)
                    blob_content = blob_client.download_blob().readall()
                    
                    content_hash = self.doc_tracker.get_document_hash(blob_content)
                    
                    if self.doc_tracker.is_document_processed(blob.name, content_hash):
                        print(f"⏭️  Already processed (unchanged): {blob.name}")
                        results["skipped_count"] += 1
                        results["skipped_documents"].append(blob.name)
                        continue
                    
                    print(f"🆕 New/changed document: {blob.name}")
                    
                    processed_result = self.doc_processor.process_blob(blob.name, blob_content)
                    
                    if not processed_result["text"].strip():
                        print(f"⚠️  No text extracted from {blob.name}")
                        continue
                    
                    print(f"📝 Extracted {len(processed_result['text'])} characters")
                    
                    print("🤖 Extracting entities and relationships...")
                    extracted_data = self.ai_extractor.extract_entities_and_relationships(
                        processed_result["text"], 
                        blob.name
                    )
                    
                    print(f"🔍 Found {len(extracted_data['entities'])} entities and {len(extracted_data['relationships'])} relationships")
                    
                    entity_ids = self._build_graph_incrementally(extracted_data, blob.name)
                    
                    processing_info = {
                        "file_type": processed_result["type"],
                        "text_length": len(processed_result["text"]),
                        "entities_found": len(extracted_data['entities']),
                        "relationships_found": len(extracted_data['relationships'])
                    }
                    
                    if "image_metadata" in processed_result:
                        processing_info["image_metadata"] = processed_result["image_metadata"]
                    
                    self.doc_tracker.mark_document_processed(
                        blob.name, 
                        content_hash, 
                        entity_ids, 
                        processing_info
                    )
                    
                    results["processed_count"] += 1
                    results["new_documents"].append({
                        "name": blob.name,
                        "entities": len(extracted_data['entities']),
                        "relationships": len(extracted_data['relationships'])
                    })
                    
                    print(f"✅ Successfully processed {blob.name}")
                    
                except Exception as e:
                    print(f"❌ Error processing {blob.name}: {e}")
                    results["error_count"] += 1
                    results["error_documents"].append({"name": blob.name, "error": str(e)})
                    continue
            
            print(f"\n🎉 INCREMENTAL PROCESSING COMPLETE!")
            print(f"📊 Summary:")
            print(f"   Total blobs checked: {results['total_blobs']}")
            print(f"   New/changed processed: {results['processed_count']}")
            print(f"   Unchanged skipped: {results['skipped_count']}")
            print(f"   Errors: {results['error_count']}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error in incremental processing: {e}")
            return {"error": str(e)}
    
    def _build_graph_incrementally(self, extracted_data: Dict[str, Any], document_name: str) -> List[str]:
        """Build knowledge graph incrementally (merge with existing)"""
        print("🏗️  Building knowledge graph incrementally...")
        
        entity_ids = []
        entity_map = {}
        
        for entity in extracted_data['entities']:
            entity_id = self.graph_builder.create_entity_vertex(entity, document_name)
            if entity_id:
                entity_ids.append(entity_id)
                entity_map[entity['name']] = entity_id
        
        for relationship in extracted_data['relationships']:
            source_name = relationship['source']
            target_name = relationship['target']
            
            source_id = entity_map.get(source_name)
            target_id = entity_map.get(target_name)
            
            if source_id and target_id:
                self.graph_builder.create_relationship_edge(relationship, source_id, target_id)
            else:
                print(f"⚠️  Could not create relationship {source_name} -> {target_name} (entities not found)")
        
        summary = extracted_data.get('document_summary', f'Document: {document_name}')
        self.graph_builder.create_document_vertex(document_name, summary, entity_ids)
        
        return entity_ids

# FIXED Streamlit UI Components
@st.cache_resource
def initialize_services():
    """Initialize all services and connections - GPT-4.1 Only Version"""
    services = {
        'status': {},'clients': {}
    }

    
    #Adding testing recommendation -v1
    st.write("🔍 Testing OpenAI connection...")
    try:
        openai_client = AzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint="https://azure-ai-foundry-genai.cognitiveservices.azure.com/",
            api_key="4VLn9irDByzTBaiFVFVKwp4dO1yZQRetUIhAEgryX1hSARocLnqwJQQJ99BEACHYHv6XJ3w3AAAAACOGegEq"
        )
        
        response = openai_client.chat.completions.create(
            model="gpt-4.1",  # This might be the wrong model name
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        services['status']['openai'] = {'status': 'connected'}
        st.write("✅ OpenAI connected!")
    except Exception as e:
        st.write(f"❌ OpenAI failed: {str(e)}")
        services['status']['openai'] = {'status': 'failed', 'error': str(e)}



    st.write("🔍 Testing Gremlin connection...")
    try:
        from gremlin_python.driver import client, serializer
        gremlin_client = client.Client(
            "wss://idefixapp-dev-graph.gremlin.cosmos.azure.com:443/",
            'g',
            username=f"/dbs/GraphDB/colls/GraphDB-id",
            password="gLEHBNYhwj15qcnz2uwNbU2DahEen9vX48EGuXZkpWIWG2xOWB3HCa9q966BGhU5Ilt05qJo9cyOACDbAQsPbA==",
            message_serializer=serializer.GraphSONSerializersV2d0()
        )
        result = gremlin_client.submit("g.V().count()")
        count = result.all().result()[0]
        services['status']['gremlin'] = {'status': 'connected', 'vertices': count}
        st.write(f"✅ Gremlin connected! {count} vertices")
    except Exception as e:
        st.write(f"❌ Gremlin failed: {str(e)}")
        services['status']['gremlin'] = {'status': 'failed', 'error': str(e)}

    # Initialize Gremlin Client
    try:

        print(f"Attempting to connect to: {Config.GREMLIN_ENDPOINT}")
        print(f"Using database: {Config.GREMLIN_DATABASE}")
        print(f"Using collection: {Config.GREMLIN_COLLECTION}")
    
        gremlin_client = client.Client(
            Config.GREMLIN_ENDPOINT,
            'g',
            username=f"/dbs/{Config.GREMLIN_DATABASE}/colls/{Config.GREMLIN_COLLECTION}",
            password=Config.GREMLIN_KEY,
            message_serializer=serializer.GraphSONSerializersV2d0()
        )
        
        result = gremlin_client.submit("g.V().count()")
        vertex_count = result.all().result()[0]
        
        services['clients']['gremlin'] = gremlin_client
        services['status']['gremlin'] = {'status': 'connected', 'vertices': vertex_count}
    except Exception as e:
        print(f"Detailed Gremlin error: {type(e).__name__}: {str(e)}")
        services['status']['gremlin'] = {'status': 'failed', 'error': str(e)}
        services['clients']['gremlin'] = None
    
    # Initialize OpenAI Client (GPT-4.1)
    try:
        openai_client = AzureOpenAI(
            api_version=Config.OPENAI_API_VERSION,
            azure_endpoint=Config.OPENAI_ENDPOINT,
            api_key=Config.OPENAI_KEY
        )
        
        response = openai_client.chat.completions.create(
            model=Config.OPENAI_DEPLOYMENT,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        
        services['clients']['openai'] = openai_client
        services['status']['openai'] = {'status': 'connected', 'model': Config.OPENAI_DEPLOYMENT}
    except Exception as e:
        services['status']['openai'] = {'status': 'failed', 'error': str(e)}
        services['clients']['openai'] = None
    
    # SKIP JAIS Client - Mark as disabled
    services['status']['jais'] = {'status': 'disabled', 'reason': 'Using GPT-4.1 for all translations'}
    services['clients']['jais'] = None
    
    # Initialize Computer Vision Client
    try:
        if COMPUTER_VISION_AVAILABLE:
            computer_vision_client = ComputerVisionClient(
                Config.COMPUTER_VISION_ENDPOINT,
                CognitiveServicesCredentials(Config.COMPUTER_VISION_KEY)
            )
            services['clients']['computer_vision'] = computer_vision_client
            services['status']['computer_vision'] = {'status': 'connected'}
        else:
            services['status']['computer_vision'] = {'status': 'failed', 'error': 'SDK not installed'}
            services['clients']['computer_vision'] = None
    except Exception as e:
        services['status']['computer_vision'] = {'status': 'failed', 'error': str(e)}
        services['clients']['computer_vision'] = None
    
    # Initialize Blob Storage Client
    try:
        blob_client = BlobServiceClient(
            account_url=f"https://{Config.STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
            credential=Config.STORAGE_ACCOUNT_KEY
        )
        
        container_client = blob_client.get_container_client(Config.CONTAINER_NAME)
        blobs = list(container_client.list_blobs())
        
        services['clients']['blob'] = blob_client
        services['status']['blob'] = {'status': 'connected', 'documents': len(blobs)}
    except Exception as e:
        services['status']['blob'] = {'status': 'failed', 'error': str(e)}
        services['clients']['blob'] = None
    
    # Initialize Azure Search Client
    try:
        search_client = SearchClient(
            endpoint=Config.SEARCH_ENDPOINT,
            index_name=Config.SEARCH_INDEX_NAME,
            credential=AzureKeyCredential(Config.SEARCH_KEY)
        )
        
        services['clients']['search'] = search_client
        services['status']['search'] = {'status': 'connected'}
    except Exception as e:
        services['status']['search'] = {'status': 'failed', 'error': str(e)}
        services['clients']['search'] = None
    
    # Initialize Translation Service - GPT-4.1 Only
    if services['clients']['openai']:
        try:
            translation_service = GPT41OnlyTranslationService(services['clients']['openai'])
            services['clients']['translation'] = translation_service
            services['status']['translation'] = {'status': 'connected', 'model': 'gpt-4.1-only'}
        except Exception as e:
            services['status']['translation'] = {'status': 'failed', 'error': str(e)}
            services['clients']['translation'] = None
    else:
        services['status']['translation'] = {'status': 'failed', 'error': 'OpenAI dependency not available'}
        services['clients']['translation'] = None
    
    # Initialize Hybrid GraphRAG System
    if services['clients']['gremlin'] and services['clients']['openai']:
        try:
            hybrid_system = HybridGraphRAGSystem(
                services['clients']['gremlin'],
                services['clients']['openai']
            )
            services['clients']['hybrid_rag'] = hybrid_system
            services['status']['hybrid_rag'] = {'status': 'connected'}
        except Exception as e:
            services['status']['hybrid_rag'] = {'status': 'failed', 'error': str(e)}
            services['clients']['hybrid_rag'] = None
    else:
        services['status']['hybrid_rag'] = {'status': 'failed', 'error': 'Dependencies not available'}
        services['clients']['hybrid_rag'] = None
    
    # Initialize Incremental Processor
    try:
        incremental_processor = IncrementalDocumentProcessor()
        services['clients']['incremental_processor'] = incremental_processor
        services['status']['incremental_processor'] = {'status': 'connected'}
    except Exception as e:
        services['status']['incremental_processor'] = {'status': 'failed', 'error': str(e)}
        services['clients']['incremental_processor'] = None
    
    return services

def render_service_status(services: Dict[str, Any]):
    """Render service status in sidebar"""
    st.sidebar.subheader("🔌 Service Status")
    
    status_map = {
        'gremlin': '🗃️ Knowledge Graph',
        'openai': '🤖 AI Engine (GPT-4.1)',
        'jais': '🤖 JAIS Arabic (Disabled)',
        'computer_vision': '👁️ Computer Vision OCR',
        'blob': '📁 Document Storage',
        'search': '🔍 Azure AI Search',
        'translation': '🌐 Translation Service',
        'hybrid_rag': '🔍 GraphRAG System',
        'incremental_processor': '🔄 Incremental Processor'
    }
    
    for service_key, service_name in status_map.items():
        status = services['status'].get(service_key, {})
        
        if status.get('status') == 'connected':
            st.sidebar.success(f"✅ {service_name}")
            
            if service_key == 'gremlin' and 'vertices' in status:
                st.sidebar.caption(f"   Vertices: {status['vertices']}")
            elif service_key == 'blob' and 'documents' in status:
                st.sidebar.caption(f"   Documents: {status['documents']}")
            elif service_key == 'openai' and 'model' in status:
                st.sidebar.caption(f"   Model: {status['model']}")
            elif service_key == 'translation' and 'model' in status:
                st.sidebar.caption(f"   Using: {status['model']}")
        elif status.get('status') == 'disabled':
            st.sidebar.info(f"ℹ️ {service_name}")
            if 'reason' in status:
                st.sidebar.caption(f"   {status['reason']}")
        else:
            st.sidebar.error(f"❌ {service_name}")
            if 'error' in status:
                st.sidebar.caption(f"   {status['error'][:50]}...")

def render_language_selector():
    """Render language selection UI in Streamlit"""
    st.sidebar.subheader("🌐 Language Settings")
    
    response_language = st.sidebar.selectbox(
        "Response Language",
        options=[lang.value for lang in SupportedLanguage],
        index=0,
        help="Choose the language for AI responses (all powered by GPT-4.1)"
    )
    
    auto_detect = st.sidebar.checkbox(
        "Auto-detect query language",
        value=True,
        help="Automatically detect the language of your questions"
    )
    
    manual_language = None
    if not auto_detect:
        manual_language = st.sidebar.selectbox(
            "Your question language",
            options=[lang.value for lang in SupportedLanguage],
            index=0,
            help="Manually specify the language of your questions"
        )
    
    return {
        "response_language": SupportedLanguage(response_language),
        "auto_detect": auto_detect,
        "manual_language": SupportedLanguage(manual_language) if manual_language else None
    }

def render_document_processor(services: Dict[str, Any]):
    """Render document processing interface"""
    st.sidebar.subheader("📚 Document Processing")
    
    if services['status']['blob']['status'] != 'connected':
        st.sidebar.warning("Blob storage not connected")
        return
    
    # Incremental processing (default)
    if st.sidebar.button("🔄 Process New/Changed Documents"):
        if services['clients']['incremental_processor']:
            with st.spinner("Processing new/changed documents..."):
                try:
                    results = services['clients']['incremental_processor'].process_new_documents()
                    
                    if "error" in results:
                        st.error(f"❌ Error: {results['error']}")
                    else:
                        st.success(f"✅ Processing complete!")
                        st.write(f"📊 **Summary:**")
                        st.write(f"- Total checked: {results['total_blobs']}")
                        st.write(f"- New/changed: {results['processed_count']}")
                        st.write(f"- Skipped: {results['skipped_count']}")
                        st.write(f"- Errors: {results['error_count']}")
                        
                        if results['new_documents']:
                            st.write("**New documents processed:**")
                            for doc in results['new_documents']:
                                st.write(f"- {doc['name']} ({doc['entities']} entities, {doc['relationships']} relationships)")
                    
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.error("Incremental processor not available")
    
    st.sidebar.caption("This processes only new or changed documents")

def render_graph_statistics(services: Dict[str, Any]):
    """Render knowledge graph statistics"""
    if services['clients']['gremlin']:
        st.sidebar.subheader("📊 Graph Statistics")
        
        try:
            gremlin_client = services['clients']['gremlin']
            
            vertex_count = gremlin_client.submit("g.V().count()").all().result()[0]
            edge_count = gremlin_client.submit("g.E().count()").all().result()[0]
            
            col1, col2 = st.sidebar.columns(2)
            col1.metric("Vertices", vertex_count)
            col2.metric("Edges", edge_count)
            
            try:
                vertex_types = gremlin_client.submit("g.V().groupCount().by(label())").all().result()[0]
                
                st.sidebar.write("**Entity Types:**")
                for entity_type, count in vertex_types.items():
                    st.sidebar.write(f"• {entity_type}: {count}")
            except Exception as e:
                st.sidebar.caption(f"Could not get vertex types: {str(e)[:50]}...")
        except Exception as e:
            st.sidebar.error(f"Error getting statistics: {str(e)}")

def process_user_query(query: str, services: Dict[str, Any], language_settings: Dict[str, Any]) -> Dict[str, Any]:
    """Process user query using hybrid GraphRAG system with translation"""
    if not services['clients']['hybrid_rag']:
        return {
            'success': False,
            'error': 'GraphRAG system not available',
            'response': 'Sorry, the GraphRAG system is not currently available. Please check the service status.'
        }
    
    try:
        # Process query with GraphRAG
        result = services['clients']['hybrid_rag'].query(query)
        
        # Apply translation if needed
        final_response = result['final_response']
        translation_info = {"translated": False}
        
        if (language_settings['response_language'] != SupportedLanguage.ENGLISH and 
            services['clients']['translation']):
            
            try:
                translation_result = services['clients']['translation'].translate_text(
                    final_response,
                    language_settings['response_language'],
                    SupportedLanguage.ENGLISH
                )
                
                final_response = translation_result["translated_text"]
                translation_info = {
                    "translated": True,
                    "model_used": translation_result["model_used"],
                    "confidence": translation_result["confidence"],
                    "original_language": SupportedLanguage.ENGLISH.value,
                    "target_language": language_settings['response_language'].value
                }
            except Exception as e:
                st.warning(f"Translation failed: {e}")
        
        return {
            'success': True,
            'result': result,
            'response': final_response,
            'original_response': result['final_response'],
            'sources': result['sources_used'],
            'translation_info': translation_info
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'response': f'An error occurred while processing your query: {str(e)}'
        }

def render_chat_interface(services: Dict[str, Any], language_settings: Dict[str, Any]):
    """Render the main chat interface"""
    st.header("💬 Knowledge Graph Chat")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show translation info if available
            if "translation_info" in message and message["translation_info"].get("translated", False):
                with st.expander("🌐 Translation Details"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Translation Model", message["translation_info"].get("model_used", "Unknown"))
                    with col2:
                        confidence = message["translation_info"].get("confidence", 0.0)
                        st.metric("Confidence", f"{confidence:.1%}")
                    
                    st.caption(f"Translated from {message['translation_info'].get('original_language', 'Unknown')} to {message['translation_info'].get('target_language', 'Unknown')}")
            
            if "metadata" in message and st.session_state.get('show_debug', False):
                with st.expander("🔍 Debug Information"):
                    st.json(message["metadata"])
    
    # Chat input
    if prompt := st.chat_input("Ask about any technology, company, or concept from your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching knowledge graph and external sources..."):
                query_result = process_user_query(prompt, services, language_settings)
                
                if query_result['success']:
                    st.markdown(query_result['response'])
                    
                    if query_result.get('sources'):
                        sources_text = " | ".join([f"📊 {source}" for source in query_result['sources']])
                        st.caption(f"Sources: {sources_text}")
                    
                    message_data = {
                        "role": "assistant",
                        "content": query_result['response'],
                        "translation_info": query_result.get('translation_info', {})
                    }
                    
                    if st.session_state.get('show_debug', False):
                        message_data["metadata"] = query_result.get('result', {})
                    
                    st.session_state.messages.append(message_data)
                else:
                    error_message = query_result['response']
                    st.error(error_message)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"❌ {error_message}"
                    })

def render_example_queries():
    """Render example queries section"""
    with st.expander("💡 Example Queries"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🔍 General Search**")
            examples = [
                "What is quantum computing?",
                "Tell me about cloud migration",
                "Azure AI services overview"
            ]
            for example in examples:
                if st.button(example, key=f"ex_{hash(example)}"):
                    st.session_state.example_query = example
                    st.rerun()
        
        with col2:
            st.markdown("**🔗 Relationships**")
            examples = [
                "How are NVIDIA and AI related?",
                "What technologies use Python?",
                "Cloud computing connections"
            ]
            for example in examples:
                if st.button(example, key=f"ex_{hash(example)}"):
                    st.session_state.example_query = example
                    st.rerun()
        
        with col3:
            st.markdown("**📊 Analysis**")
            examples = [
                "Compare OLAP systems",
                "Mainframe migration strategies", 
                "AI implementation best practices"
            ]
            for example in examples:
                if st.button(example, key=f"ex_{hash(example)}"):
                    st.session_state.example_query = example
                    st.rerun()

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Comprehensive Knowledge Graph",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .source-badge {
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 4px 8px;
        border-radius: 15px;
        font-size: 0.8em;
        margin: 2px;
        display: inline-block;
    }
    
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Comprehensive Dynamic Knowledge Graph System</h1>
        <p>Intelligent document processing with Image OCR, Incremental Updates, Multi-Language Translation (GPT-4.1), and GraphRAG</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize services
    if 'services' not in st.session_state:
        with st.spinner("🚀 Initializing comprehensive services..."):
            st.session_state.services = initialize_services()
    
    services = st.session_state.services
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ System Control")
        
        # Service status
        render_service_status(services)
        
        # Language settings
        language_settings = render_language_selector()
        
        # Document processing
        render_document_processor(services)
        
        # Graph statistics
        render_graph_statistics(services)
        
        # Settings
        st.subheader("🛠️ Settings")
        st.session_state.show_debug = st.checkbox("Show debug information", value=False)
        
        if st.button("🔄 Refresh Services"):
            del st.session_state.services
            st.rerun()
        
        # System capabilities info
        st.subheader("🚀 System Capabilities")
        capabilities = [
            "📄 Text Documents (PDF, DOCX, TXT)",
            "🖼️ Image OCR (PNG, JPG, etc.)",
            "🔄 Incremental Processing", 
            "🌐 Multi-Language Translation (GPT-4.1)",
            "🔍 Hybrid GraphRAG Querying",
            "📊 Azure AI Search Integration",
            "🗃️ Dynamic Knowledge Graph"
        ]
        
        for capability in capabilities:
            st.caption(capability)
    
    # Main content
    if services['status']['hybrid_rag']['status'] == 'connected':
        render_chat_interface(services, language_settings)
        render_example_queries()
    else:
        st.error("❌ GraphRAG system is not available. Please check service status and process documents first.")
        
        if services['status']['gremlin']['status'] == 'connected':
            if services['status']['gremlin'].get('vertices', 0) == 0:
                st.info("💡 Your knowledge graph appears to be empty. Try processing your documents first using the sidebar.")
        
        st.subheader("🔧 Troubleshooting")
        st.write("""
        1. **Check Service Status**: Look at the sidebar to see which services are connected
        2. **Process Documents**: Use the "Process New/Changed Documents" button to build your knowledge graph
        3. **Verify Configuration**: Make sure your Azure credentials are correct
        4. **Check Storage**: Ensure your blob storage contains documents to process
        5. **Translation**: All languages now use GPT-4.1 (JAIS disabled as not deployed)
        """)

if __name__ == "__main__":

    main()

