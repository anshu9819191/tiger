from flask import Flask, render_template, request, send_file, redirect, url_for
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import io
import os
from datetime import datetime

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8MB per request (adjust if needed)

mp_face_mesh = mp.solutions.face_mesh

def read_image_file(file_storage):
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img

def get_landmarks(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        res = face_mesh.process(img_rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        h, w = img_bgr.shape[:2]
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]
        return pts

def convex_hull_pts(landmarks, idx_list):
    return np.array([landmarks[i] for i in idx_list], np.int32)

# A set of indices roughly covering face from MediaPipe face mesh (use a broad set)
FACE_IDX = list(range(10, 338))  # wide range; will be filtered by convex hull

def seamless_swap(src, dst):
    """Swap src face into dst image using Mediapipe landmarks & OpenCV seamlessClone."""
    src_pts = get_landmarks(src)
    dst_pts = get_landmarks(dst)
    if src_pts is None or dst_pts is None:
        return None, "Couldn't detect face in one of the images. Try clearer, frontal faces."

    # Use convex hull of facial landmarks to create face mask
    src_h, src_w = src.shape[:2]
    dst_h, dst_w = dst.shape[:2]

    # pick landmarks that roughly cover the face: use all and compute hull
    src_pts_arr = np.array(src_pts, np.int32)
    dst_pts_arr = np.array(dst_pts, np.int32)

    src_hull = cv2.convexHull(src_pts_arr)
    dst_hull = cv2.convexHull(dst_pts_arr)

    # Create mask for source face
    src_mask = np.zeros((src_h, src_w), dtype=np.uint8)
    cv2.fillConvexPoly(src_mask, src_hull, 255)

    # bounding rect for src face
    x, y, w, h = cv2.boundingRect(src_hull)
    center = (x + w // 2, y + h // 2)

    # warp source face onto destination using affine transforms based on 3 landmark points
    # choose left eye outer, right eye outer, and nose tip indices (approx positions in MediaPipe)
    idx_pts = [33, 263, 1]  # left eye outer, right eye outer, nose tip
    src_tri = np.float32([src_pts[i] for i in idx_pts])
    dst_tri = np.float32([dst_pts[i] for i in idx_pts])

    # compute affine transform
    M = cv2.getAffineTransform(src_tri, dst_tri)

    warped_src = cv2.warpAffine(src, M, (dst_w, dst_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    warped_mask = cv2.warpAffine(src_mask, M, (dst_w, dst_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # compute center of dst hull for seamlessClone
    x2, y2, w2, h2 = cv2.boundingRect(dst_hull)
    center_dst = (x2 + w2 // 2, y2 + h2 // 2)

    # convert to 3 channels mask
    warped_mask_3 = cv2.merge([warped_mask, warped_mask, warped_mask])

    # prepare images for clone
    try:
        output = cv2.seamlessClone(warped_src, dst, warped_mask, center_dst, cv2.NORMAL_CLONE)
    except Exception as e:
        # fallback: if seamlessClone fails, blend directly
        inv_mask = cv2.bitwise_not(warped_mask)
        bg = cv2.bitwise_and(dst, dst, mask=inv_mask)
        fg = cv2.bitwise_and(warped_src, warped_src, mask=warped_mask)
        output = cv2.add(bg, fg)

    return output, None

def draw_stylish_text(cv_img, text):
    # Convert to PIL for nicer text rendering with fonts, gradients etc.
    pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)).convert("RGBA")
    draw = ImageDraw.Draw(pil)

    # choose font size relative to image width
    w, h = pil.size
    font_size = max(20, w // 12)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        # fallback to default PIL font
        font = ImageFont.load_default()

    # prepare text position: bottom center with padding
    padding = int(h * 0.03)
    text_w, text_h = draw.textsize(text, font=font)
    x = (w - text_w) // 2
    y = h - text_h - padding

    # Create shadow / outline for pop effect
    outline_range = 2
    for ox in range(-outline_range, outline_range+1):
        for oy in range(-outline_range, outline_range+1):
            draw.text((x+ox, y+oy), text, font=font, fill=(0,0,0,160))

    # Simple multi-color gradient fill: draw each character with shifted hue
    # We'll pick a palette of colors and cycle through
    palette = [(255, 102, 102, 255), (255, 178, 102, 255), (255, 255, 102, 255),
               (178, 255, 102, 255), (102, 255, 178, 255), (102, 178, 255, 255),
               (178, 102, 255, 255), (255, 102, 229, 255)]
    tx = x
    for i, ch in enumerate(text):
        ch_w, ch_h = draw.textsize(ch, font=font)
        color = palette[i % len(palette)]
        draw.text((tx, y), ch, font=font, fill=color)
        tx += ch_w

    # Add a subtle translucent rectangle behind the text
    rect_h = text_h + padding//2
    rect_y = y - padding//3
    overlay = Image.new('RGBA', pil.size, (255,255,255,0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([(x - padding, rect_y), (x + text_w + padding, rect_y + rect_h)],
                           fill=(0,0,0,60))
    pil = Image.alpha_composite(pil, overlay)

    return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img1 = request.files.get('img1')
        img2 = request.files.get('img2')
        text = request.form.get('overlay_text', '').strip()

        if not img1 or not img2:
            return render_template('index.html', error="Donor dono images upload karein.", text=text)

        try:
            src = read_image_file(img1)
            dst = read_image_file(img2)
        except Exception as e:
            return render_template('index.html', error="Uploaded files ko read karne mein error: " + str(e), text=text)

        swapped, err = seamless_swap(src, dst)
        if err:
            return render_template('index.html', error=err, text=text)

        if text:
            swapped = draw_stylish_text(swapped, text)

        # encode to JPEG and send back as downloadable image (in-memory)
        is_success, buffer = cv2.imencode(".jpg", swapped, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not is_success:
            return render_template('index.html', error="Result encode nahi hua.", text=text)

        io_buf = io.BytesIO(buffer.tobytes())
        filename = f"face_swap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        return send_file(io_buf, mimetype='image/jpeg', as_attachment=False, download_name=filename)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
