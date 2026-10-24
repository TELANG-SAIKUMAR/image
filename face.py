"""
Photo -> textured 3D head (GLB), no AI model downloads needed.

How it works (plain English):
  1. Find the head+hair outline in the photo (the background is plain grey).
  2. Build a closed, head-shaped 3D surface that follows that outline.
  3. Push the nose, lips, brows, cheeks and chin out (and the eye sockets in)
     at the spots where they are in the photo.
  4. Wrap the photo on the FRONT of the head, and a matching hair-colour
     patch on the BACK so it doesn't look hollow from behind.
  5. Save as a small GLB (JPEG texture, < 2 MB).

Usage:  python make_face_glb.py input.jpg out.glb
"""
import sys, struct, json, io
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d

SRC = sys.argv[1] if len(sys.argv) > 1 else "input.jpg"
OUT = sys.argv[2] if len(sys.argv) > 2 else "face.glb"
MAX_BYTES = 2_000_000

# ---- landmark positions in the 1280x1920 reference photo (pixels) -------------
# If you use a different photo size, these scale automatically below.
REF_W, REF_H = 1280, 1920
LM = dict(
    eyeL=(530, 445), eyeR=(700, 440),
    nose_top=(605, 430), nose_tip=(603, 568),
    mouth=(603, 675), chin=(603, 795),
    brow_y=385, brow_x=(440, 780),
    cheekL=(480, 595), cheekR=(735, 595),
)

img = Image.open(SRC).convert("RGB")
W, H = img.size
sx, sy = W / REF_W, H / REF_H
arr = np.asarray(img).astype(np.float32)

def P(pt):  # scale a landmark to this image's size
    return (pt[0] * sx, pt[1] * sy)

# ---- 1. head + hair outline from the plain background ------------------------
bg = np.median(np.concatenate([arr[:40, :60].reshape(-1, 3), arr[:40, -60:].reshape(-1, 3)]), axis=0)
dist = np.linalg.norm(arr - bg, axis=2)
mask = gaussian_filter(dist, 2) > 38

Y_TOP = int(95 * sy)
Y_BOT = int(900 * sy)           # where the head/hair block ends
rows = np.arange(Y_TOP, Y_BOT)
L = np.zeros(len(rows)); R = np.zeros(len(rows))
for i, y in enumerate(rows):
    xs = np.where(mask[y])[0]
    xs = xs[(xs > W * 0.12) & (xs < W * 0.88)]
    if len(xs) == 0:
        L[i] = R[i] = W / 2
    else:
        L[i], R[i] = xs.min(), xs.max()
# first rows of hair can be empty -> fill by interpolation
good = (R - L) > 40
L = np.interp(rows, rows[good], L[good]); R = np.interp(rows, rows[good], R[good])
L = gaussian_filter1d(L, 12); R = gaussian_filter1d(R, 12)
INSET = 28 * sx
grow = 1 + 2.2 * np.clip((rows - 560 * sy) / (340 * sy), 0, 1)   # pull in more where hair gets wispy
L += INSET * grow; R -= INSET * grow
cx_y = (L + R) / 2
w_y = (R - L) / 2

# round the top (outline at the very top is too pointy/noisy)
top_n = int(60 * sy)
t = np.linspace(0, 1, top_n)
w_y[:top_n] *= np.sqrt(np.clip(1 - (1 - t) ** 2, 0, 1)) * 0.0 + np.sin(t * np.pi / 2) ** 0.7
# round the bottom (closing cap)
bot_n = int(110 * sy)
t = np.linspace(0, 1, bot_n)
w_y[-bot_n:] *= np.sqrt(np.clip(1 - t ** 2, 0.02, 1))

# ---- 2. grid over (height, angle) --------------------------------------------
NR, NC = 150, 96                 # rows x columns per half (front / back)
ys = np.linspace(rows[0], rows[-1], NR)
w = np.interp(ys, rows, w_y); cx = np.interp(ys, rows, cx_y)
w = gaussian_filter1d(w, 1.5)

K = 0.74                                   # depth / width of the head
d_back = K * w * 0.95
d_front = K * w
taper = np.clip((ys - 800 * sy) / (110 * sy), 0, 1)
d_front = d_front * (1 - 0.5 * taper ** 1.5)   # neck sits behind the chin

th_front = np.linspace(-np.pi / 2, np.pi / 2, NC)
th_back = np.linspace(np.pi / 2, 3 * np.pi / 2, NC)

def gauss(X, Y, c, sxp, syp, amp):
    return amp * np.exp(-(((X - c[0]) / sxp) ** 2 + ((Y - c[1]) / syp) ** 2) / 2)

def face_relief(X, Y):
    """extra depth (pixels) in front of the base head, at photo coords X,Y"""
    z = np.zeros_like(X)
    nt, nb = P(LM["nose_top"]), P(LM["nose_tip"])
    # nose bridge: ramps up from nose_top to nose_tip
    tt = np.clip((Y - nt[1]) / (nb[1] - nt[1]), 0, 1)
    ridge = (12 + 32 * tt ** 1.6) * np.exp(-((X - nb[0]) / (24 * sx * (1 + 0.6 * tt))) ** 2 / 2 * 2)
    ridge *= ((Y > nt[1] - 10) & (Y < nb[1] + 20)).astype(float) * np.clip((nb[1] + 20 - Y) / 25, 0, 1)
    z += gaussian_filter(ridge, 1.0)
    z += gauss(X, Y, nb, 30 * sx, 24 * sy, 32)                       # tip
    z += gauss(X, Y, (nb[0] - 33 * sx, nb[1] + 2 * sy), 17 * sx, 15 * sy, 18)   # nostril wings
    z += gauss(X, Y, (nb[0] + 33 * sx, nb[1] + 2 * sy), 17 * sx, 15 * sy, 18)
    for e in (LM["eyeL"], LM["eyeR"]):                               # eye sockets
        z += gauss(X, Y, P(e), 46 * sx, 26 * sy, -16)
    bx0, bx1 = LM["brow_x"]                                          # brow ridge
    z += gauss(X, Y, P(((bx0 + bx1) / 2, LM["brow_y"])), (bx1 - bx0) / 2.6 * sx, 18 * sy, 14)
    z += gauss(X, Y, P(LM["cheekL"]), 60 * sx, 55 * sy, 14)          # cheeks
    z += gauss(X, Y, P(LM["cheekR"]), 60 * sx, 55 * sy, 14)
    m = P(LM["mouth"])
    z += gauss(X, Y, m, 62 * sx, 20 * sy, 26)                        # lips
    z += gauss(X, Y, (m[0], m[1] + 62 * sy), 34 * sx, 22 * sy, -10)  # dip under lip
    z += gauss(X, Y, (m[0], m[1] - 45 * sy), 50 * sx, 18 * sy, -6)   # above lip
    z += gauss(X, Y, P(LM["chin"]), 52 * sx, 34 * sy, 24)            # chin
    return z

def build_half(theta, front):
    TH, YY = np.meshgrid(theta, ys)          # (NR, NC)
    CX = cx[:, None]; WW = w[:, None]
    X = CX + WW * np.sin(TH)
    c = np.cos(TH)
    if front:
        Z = d_front[:, None] * c
        Z = Z + face_relief(X, YY) * np.clip(c, 0, 1) ** 0.6
    else:
        Z = d_back[:, None] * c               # c<0 here
    return X, YY, Z

Xf, Yf, Zf = build_half(th_front, True)
Xb, Yb, Zb = build_half(th_back, False)
# smooth the front a touch so bumps are not jagged
Zf_s = gaussian_filter(Zf, 0.8)
Zf = Zf_s

# ---- 3. texture atlas: photo (front) + hair patch (back) ---------------------
CROP = (int(150 * sx), int(60 * sy), int(1130 * sx), int(960 * sy))   # photo region used
cw, ch = CROP[2] - CROP[0], CROP[3] - CROP[1]
TW_F, TH_ = 1024, 1024
PATCH = 256
atlas = Image.new("RGB", (TW_F + PATCH, TH_))
atlas.paste(img.crop(CROP).resize((TW_F, TH_), Image.LANCZOS), (0, 0))

# hair patch: colour at the left/right edge of each row, blended across, with strand streaks
rng = np.random.default_rng(3)
patch = np.zeros((TH_, PATCH, 3), np.float32)
for r in range(TH_):
    ypix = CROP[1] + r * ch / TH_
    i = int(np.clip(np.searchsorted(rows, ypix), 0, len(rows) - 1))
    xl = cx_y[i] - w_y[i]; xr = cx_y[i] + w_y[i]
    def edge(x0, sign):
        xs_ = np.clip(np.arange(x0, x0 + sign * 30, sign).astype(int), 0, W - 1)
        return arr[int(ypix), xs_].mean(0)
    cl, cr = edge(xl + 4 * sx, 1), edge(xr - 4 * sx, -1)
    tcol = np.linspace(0, 1, PATCH)[:, None]
    # left edge -> darker/warmer middle -> right edge
    mid = (cl + cr) / 2 * 0.92
    left = cl * (1 - tcol * 2).clip(0, 1) + mid * (tcol * 2).clip(0, 1)
    right = mid * (1 - (tcol - .5) * 2).clip(0, 1) + cr * ((tcol - .5) * 2).clip(0, 1)
    patch[r] = np.where(tcol < .5, left, right)
streak = gaussian_filter(rng.normal(size=(TH_, PATCH)), (40, 1.2))
streak = streak / (np.abs(streak).max() + 1e-6)
patch *= (1 + 0.10 * streak)[..., None]
patch = gaussian_filter(patch, (0, 0.6, 0))
atlas.paste(Image.fromarray(np.clip(patch, 0, 255).astype(np.uint8)), (TW_F, 0))
ATW = TW_F + PATCH

def uv_front(X, Y):
    u = (X - CROP[0]) / cw * (TW_F / ATW)
    v = (Y - CROP[1]) / ch
    return u, v

def uv_back(theta, rows_n):
    s = (theta - np.pi / 2) / np.pi                      # 0..1 across the back
    u = (TW_F + 4 + s * (PATCH - 8)) / ATW
    v = (ys - CROP[1]) / ch
    return np.broadcast_to(u, (rows_n, len(theta))).copy(), np.broadcast_to(v[:, None], (rows_n, len(theta))).copy()

# ---- 4. assemble vertices / normals / faces ----------------------------------
# glTF axes: +X right, +Y up, +Z toward viewer. Scale so the head is ~1.8 units tall.
SCALE = 1.8 / (rows[-1] - rows[0])
def to3d(X, Y, Z):
    return np.stack([(X - W / 2) * SCALE, -(Y - (rows[0] + rows[-1]) / 2) * SCALE, Z * SCALE], -1)

Pf, Pb = to3d(Xf, Yf, Zf), to3d(Xb, Yb, Zb)

def normals(Pg, flip):
    dx = np.gradient(Pg, axis=1); dy = np.gradient(Pg, axis=0)
    n = np.cross(dx, dy)
    n /= (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9)
    return -n if flip else n

# make the normals face outward (check on the front-centre vertex)
nf = normals(Pf, False)
if nf[NR // 2, NC // 2, 2] < 0: nf = -nf
nb_ = normals(Pb, False)
if nb_[NR // 2, NC // 2, 2] > 0: nb_ = -nb_
# smooth normals across the seam (average duplicated seam vertices)
for a, b in ((nf[:, 0], nb_[:, -1]), (nf[:, -1], nb_[:, 0])):
    m = a + b; m /= np.linalg.norm(m, axis=-1, keepdims=True) + 1e-9
    a[:] = m; b[:] = m

uf, vf = uv_front(Xf, Yf)
ub, vb = uv_back(th_back, NR)

def grid_faces(nr, nc, off, flip):
    idx = np.arange(nr * nc).reshape(nr, nc) + off
    a, b, c_, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    f1 = np.stack([a, c_, b], -1).reshape(-1, 3); f2 = np.stack([b, c_, d], -1).reshape(-1, 3)
    f = np.concatenate([f1, f2])
    return f[:, ::-1] if flip else f

# decide winding so triangles face outward: test on the front
def winding_ok(Pg, faces, n):
    p = Pg.reshape(-1, 3)
    tri = p[faces - faces.min()]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return (fn * n.reshape(-1, 3)[faces[:, 0] - faces.min()]).sum(1).mean() > 0

ff = grid_faces(NR, NC, 0, False)
if not winding_ok(Pf, ff, nf): ff = grid_faces(NR, NC, 0, True)
fb = grid_faces(NR, NC, NR * NC, False)
if not winding_ok(Pb, fb, nb_): fb = grid_faces(NR, NC, NR * NC, True)

pos = np.concatenate([Pf.reshape(-1, 3), Pb.reshape(-1, 3)]).astype(np.float32)
nor = np.concatenate([nf.reshape(-1, 3), nb_.reshape(-1, 3)]).astype(np.float32)
uvs = np.concatenate([np.stack([uf, vf], -1).reshape(-1, 2), np.stack([ub, vb], -1).reshape(-1, 2)]).astype(np.float32)
idx = np.concatenate([ff, fb]).astype(np.uint16)

# ---- 5. write GLB by hand (JPEG texture keeps it small) ----------------------
def write_glb(path, quality):
    buf = io.BytesIO(); atlas.save(buf, "JPEG", quality=quality, optimize=True, progressive=False)
    jpg = buf.getvalue()
    chunks, views = [], []
    def add(data, target=None):
        pad = (-sum(len(c) for c in chunks)) % 4
        if pad: chunks.append(b"\0" * pad)
        off = sum(len(c) for c in chunks)
        chunks.append(data)
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        if target: v["target"] = target
        views.append(v); return len(views) - 1
    vp = add(pos.tobytes(), 34962); vn = add(nor.tobytes(), 34962)
    vu = add(uvs.tobytes(), 34962); vi = add(idx.tobytes(), 34963); vj = add(jpg)
    gltf = {
        "asset": {"version": "2.0", "generator": "make_face_glb.py"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": "head"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
                                    "indices": 3, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0},
                                                "metallicFactor": 0.0, "roughnessFactor": 0.9},
                       "doubleSided": True}],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "images": [{"bufferView": vj, "mimeType": "image/jpeg"}],
        "accessors": [
            {"bufferView": vp, "componentType": 5126, "count": len(pos), "type": "VEC3",
             "min": pos.min(0).tolist(), "max": pos.max(0).tolist()},
            {"bufferView": vn, "componentType": 5126, "count": len(nor), "type": "VEC3"},
            {"bufferView": vu, "componentType": 5126, "count": len(uvs), "type": "VEC2"},
            {"bufferView": vi, "componentType": 5123, "count": len(idx.reshape(-1)), "type": "SCALAR"}],
        "bufferViews": views,
    }
    binb = b"".join(chunks); binb += b"\0" * ((-len(binb)) % 4)
    gltf["buffers"] = [{"byteLength": len(binb)}]
    js = json.dumps(gltf, separators=(",", ":")).encode(); js += b" " * ((-len(js)) % 4)
    total = 12 + 8 + len(js) + 8 + len(binb)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(js), b"JSON")); f.write(js)
        f.write(struct.pack("<I4s", len(binb), b"BIN\0")); f.write(binb)
    return total

for q in (90, 85, 80, 75, 70, 60):
    size = write_glb(OUT, q)
    print(f"JPEG quality {q}: {size:,} bytes")
    if size <= 1_800_000:
        break
print("OK, under 2,000,000 bytes" if size <= MAX_BYTES else "STILL TOO BIG - lower NR/NC or TW_F")