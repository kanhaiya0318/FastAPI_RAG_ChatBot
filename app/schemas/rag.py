from pydantic import BaseModel, ConfigDict

class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    doc_id: str
    source: str
    total_pages: int
    chunks: int

class UploadOut(DocumentOut):
    skipped: bool   # True if this exact file was already indexed

class DeleteOut(BaseModel):
    doc_id: str
    deleted: bool