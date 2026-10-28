from fastapi import APIRouter, HTTPException, status, Depends
from app.dependancies import get_db
from app.schemas.chat import HistoryResponse, EndSessionResponse


router = APIRouter(
    prefix = "/sessions",
    tags = ["session"]
)

@router.post("/{session_id}/end", status_code = status.HTTP_200_OK)
async def end_chat_session(session_id :str, db: dict = Depends(get_db)):

    if session_id in db:
        del db[session_id]
        return EndSessionResponse(status = "success", detail= f"session {session_id} is deleted.")

    raise HTTPException(
        status_code = status.HTTP_404_NOT_FOUND,
        detail = "session is already deleted or not found."
    )


@router.get("/history/{session_id}", response_model = HistoryResponse)
async def get_history(session_id:str, db:dict = Depends(get_db)) -> dict:

    if session_id not in db:
        raise HTTPException(
            status_code= status.HTTP_404_NOT_FOUND,
            detail = "History is not found."
        )
    return HistoryResponse(session_id= session_id, summary= db[session_id]["summary"], history= db[session_id]["history"])


@router.get("/get_session_ids", status_code = status.HTTP_200_OK)
def get_session_ids(db:dict = Depends(get_db))->list:
    return list(db.keys())

