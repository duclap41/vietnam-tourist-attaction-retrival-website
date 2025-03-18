from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import shutil
import os
from pathlib import Path
import uuid
import base64

from retrieval.feature_extraction import init_pinecone, get_top_k_img, get_top_k_img_with_filter,  get_top_k_img_with_weights, feedback_manager
from description import get_wikipedia_content
from collections import Counter

# ============= Constants & Configurations =============
app = FastAPI()

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")
# Set up templates
templates = Jinja2Templates(directory="templates")

IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif"}
UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)



map_description = {
    0: "Lăng Chủ tịch Hồ Chí Minh",
    1: "Quần thể danh thắng Tràng An",
    2: "Kinh thành Huế",
    3: "Nhà thờ chính tòa Đức Bà Sài Gòn",
    4: "Vịnh Hạ Long",
    5: "Dinh Độc Lập",
    6: "Thánh địa Mỹ Sơn",
    7: "Hồ Hoàn Kiếm",
    8: "Bưu điện Sài Gòn",
    9: "Tượng Chúa Kitô Vua (Vũng Tàu)",
    10: "Cầu Vàng",
    11: "Cột cờ Lũng Cú",
    12: "Động Phong Nha",
    13: "Phố cổ Hội An",
    14: "Thác Bản Giốc",
    15: "Nhà thờ Lớn Hà Nội",
    16: "Chợ Bến Thành",
    17: "Ga Đà Lạt",
    18: "Chợ nổi",
    19: "Gành Đá Đĩa"
}

# ============= Global State =============

# All query images
current_queries = []
retrieval_results = []
description = ""
wiki_link = ""
current_model = None
results_top40 = []

# ============= Helper Functions =============

# ============= Route Handlers =============
# Homepage Process
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "images": current_queries,
        }
    )


@app.post("/uploads/")
async def upload_images(request: Request, file: UploadFile = File(...)):
    # Check type file
    if file.content_type not in IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="Supported file type (JPEG, PNG, GIF)")

    await file.seek(0)
    original_extension = Path(file.filename).suffix
    unique_filename = f"{str(uuid.uuid4())[:8]}{original_extension}"

    file_path = f"{UPLOAD_DIR}/{unique_filename}"

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Add to uploaded images
        current_queries.append(unique_filename)

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "filename": unique_filename,
                "images": current_queries,
            })

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=500,
            detail=f"Uploading Error: {str(e)}"
        )


@app.post("/delete/{filename}")
async def delete_file(request: Request, filename: str):
    try:
        file_path = f"{UPLOAD_DIR}/{filename}"
        if os.path.exists(file_path):
            os.remove(file_path)
            current_queries.remove(filename)

        return RedirectResponse(
            url="/",
            status_code=303  # 303 See Other
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Delete error: {str(e)}"
        )


# Lead to enhance section
@app.post("/enhance/")
async def enhance(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "images": current_queries,
            "allow_enhance": True
        })


# Thêm route GET cho enhance
@app.get("/enhance/")
async def enhance_get(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "images": current_queries,
            "allow_enhance": True
        })

# Run retrieval
@app.get("/search/", response_class=HTMLResponse)
async def search(request: Request, retrieve_model: str):
    global results_top40, current_model, del_count
    try:
        # Reset các biến global
        current_model = retrieve_model
        del_count = 0
        results_top40.clear()

        # Reset feedback manager cho lần search mới
        feedback_manager.deleted_images.clear()
        feedback_manager.reranked_positions.clear()

        index = init_pinecone(f'{retrieve_model}-index')
        temp_results = []
        meta_data_list = []

        if len(current_queries) == 1:
            query_path = f"{UPLOAD_DIR}/{current_queries[0]}"
            try:
                # Sử dụng weights đã học được từ các lần trước
                temp_data = get_top_k_img_with_weights(
                    query_img_path=query_path,
                    index=index,
                    model_type=retrieve_model,
                    top_k=40
                )

                # Lưu top 40 kết quả
                results_top40 = [(int(item['metadata']['img_class']), item['metadata']['img_path'])
                                 for item in sorted(temp_data, key=lambda x: x['score'], reverse=True)]

                # Sử dụng 10 kết quả đầu cho hiển thị
                temp_results = temp_data[:10]

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

        else:
            # Multiple query processing
            for coef in range(len(current_queries), 0, -1):
                query_path = f"{UPLOAD_DIR}/{current_queries[len(current_queries) - coef]}"
                top_k = coef * 10 if coef == 1 else coef * 20

                try:
                    if coef == len(current_queries):
                        # Initial search với weights
                        temp_results = get_top_k_img_with_weights(
                            query_img_path=query_path,
                            index=index,
                            model_type=retrieve_model,
                            top_k=top_k
                        )
                    else:
                        # Filtered search với weights
                        temp_results = get_top_k_img_with_filter(
                            query_img_path=query_path,
                            index=index,
                            top_k=top_k,
                            model=retrieve_model,
                            filter_list=meta_data_list,
                            include_value=False
                        )

                    meta_data_list = [match['metadata']['img_path'] for match in temp_results]

                    if coef == 2:
                        results_top40 = [(int(item['metadata']['img_class']), item['metadata']['img_path'])
                                         for item in sorted(temp_results, key=lambda x: x['score'], reverse=True)]

                except Exception as e:
                    continue

        # Update retrieval results
        retrieval_results.clear()
        sorted_data = sorted(temp_results, key=lambda x: x['score'], reverse=True)
        retrieval_results.extend([(int(item['metadata']['img_class']), item['metadata']['img_path'])
                                  for item in sorted_data])

        # Update class prediction and description
        class_counts = Counter([result[0] for result in retrieval_results])
        most_common_class, _ = class_counts.most_common(1)[0]

        global description, wiki_link
        description, wiki_link = get_wikipedia_content(most_common_class, lang='vi')

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "search_images": retrieval_results,
                "predict_class": map_description[most_common_class],
                "description": description,
                "wiki_link": wiki_link,
                "start_search": True,
                "images": current_queries,
                "allow_enhance": True
            })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

del_count = 0


@app.post("/search/delete/top{k}")
async def delete_topk(request: Request, k: int):
    try:
        global del_count, results_top40

        # Validation checks
        if not results_top40:
            raise ValueError("No search results available")
        if k - 1 < 0 or k - 1 >= len(retrieval_results):
            raise ValueError("Invalid index for deletion")
        if 10 + del_count >= len(results_top40):
            raise ValueError("No more results to add")

        # Get and remove the deleted image
        deleted_image = retrieval_results.pop(k - 1)
        del_count += 1

        try:
            # Update weights based on deletion feedback
            feedback_manager.update_weights_from_deletion(
                img_path=deleted_image[1],
                model_type=current_model
            )

            # Requery với weights mới
            query_path = f"{UPLOAD_DIR}/{current_queries[0]}"
            index = init_pinecone(f'{current_model}-index')

            # Get new results with updated weights
            new_results = get_top_k_img_with_weights(
                query_img_path=query_path,
                index=index,
                model_type=current_model,
                top_k=40 - del_count
            )

            # Update results_top40 với kết quả mới
            results_top40 = [(int(item['metadata']['img_class']), item['metadata']['img_path'])
                             for item in sorted(new_results, key=lambda x: x['score'], reverse=True)]

            # Thêm kết quả tiếp theo vào retrieval_results
            retrieval_results.append(results_top40[10 + del_count - 1])

            # Update class prediction
            class_counts = Counter([result[0] for result in retrieval_results])
            most_common_class, _ = class_counts.most_common(1)[0]

            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "search_images": retrieval_results,
                    "predict_class": map_description[most_common_class],
                    "description": description,
                    "wiki_link": wiki_link,
                    "start_search": True,
                    "images": current_queries,
                }
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete error: {str(e)}")


@app.post("/search/reorder")
async def reorder_rank(request: Request):
    try:
        data = await request.json()
        new_order = data.get('new_order', [])

        # Cập nhật thứ tự hiện tại
        global retrieval_results
        original_images = retrieval_results.copy()
        retrieval_results = [original_images[int(index)] for index in new_order]

        # Tạo dictionary cho reranking feedback
        reranked_positions = {
            img[1]: new_idx
            for new_idx, img in enumerate(retrieval_results)
        }

        # Cập nhật weights
        feedback_manager.update_weights_from_reranking(
            reranked_images=reranked_positions,
            model_type=current_model
        )

        # Requery với weights mới
        query_path = f"{UPLOAD_DIR}/{current_queries[0]}"
        index = init_pinecone(f'{current_model}-index')

        new_results = get_top_k_img_with_weights(
            query_img_path=query_path,
            index=index,
            model_type=current_model,
            top_k=40 - del_count
        )

        # Cập nhật results_top40
        global results_top40
        results_top40 = [(int(item['metadata']['img_class']), item['metadata']['img_path'])
                         for item in sorted(new_results, key=lambda x: x['score'], reverse=True)]

        return {
            "status": "success",
            "message": "Reordered and weights updated successfully",
            "new_results": results_top40[:10]  # Trả về 10 kết quả đầu tiên với weights mới
        }

    except Exception as e:
        print(f"Error in reorder_rank: {e}")
        return {"status": "error", "message": str(e)}


# Editing Ground
@app.get("/editing-ground/{filename}", response_class=HTMLResponse)
async def edit_ground(request: Request, filename: str):
    return templates.TemplateResponse("edit-ground.html", {
        "request": request,
        "filename": filename
    })


@app.post("/save-image/{filename}")
async def save_image(request: Request, filename: str, image_data: dict):
    try:
        # Remove header của base64 string
        image_data_str = image_data.get("image_data", "")
        if "base64," in image_data_str:
            image_data_str = image_data_str.split("base64,")[1]

        # Decode base64 thành binary
        image_binary = base64.b64decode(image_data_str)

        # Lưu đè lên file cũ với cùng tên file
        file_path = f"static/uploads/{filename}"
        with open(file_path, "wb") as f:
            f.write(image_binary)

        # Redirect to enhance route với cùng danh sách ảnh
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "images": current_queries,  # Giữ nguyên danh sách ảnh hiện tại
                "allow_enhance": True
            }
        )
    except Exception as e:
        print(f"Save image error: {str(e)}")
        return {"success": False, "error": str(e)}


@app.get("/cancel-edit")
async def cancel_edit(request: Request):
    # Redirect to enhance route
    return RedirectResponse(url="/enhance/", status_code=303)

from auto_enhance import auto_enhancing
import cv2

@app.post("/auto-enhance/{filename}")
async def auto_enhance(request: Request, filename: str):
    try:
        file_path = f"static/uploads/{filename}"

        # Execute Auto Enhance Quality
        enhanced_img = auto_enhancing(file_path)

        enhanced_img_bgr = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)

        # Encode to base64
        _, buffer = cv2.imencode('.png', enhanced_img_bgr)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "success": True,
            "image": f"data:image/png;base64,{img_base64}"
        }

    except Exception as e:
        print(f"Auto enhance error: {str(e)}")
        return {"success": False, "error": str(e)}