"""Create original vector schematics for the YOLOv1 course report.

Architectural/algorithmic content is redrawn from Redmon et al. (CVPR 2016).
Schematics are explanatory drawings, not experimental detector outputs.
"""
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/yolo-report-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "figures"
BLUE, INK, PALE, TEAL, ORANGE = "#1F4E79", "#243448", "#EDF3F8", "#237F88", "#A55C31"
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9, "pdf.fonttype":42})


def canvas(height):
    fig, ax = plt.subplots(figsize=(6.65, height))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, height / 6.65 * 10)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, color=BLUE, fill=PALE, fs=9):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.04,rounding_size=0.08",
                              linewidth=0.9, edgecolor=color, facecolor=fill))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs,color=INK,linespacing=1.35)


def arrow(ax, start, end, color=BLUE):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=10,
                               linewidth=1,color=color,shrinkA=3,shrinkB=3))


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def pipelines():
    fig, ax = canvas(2.75)
    ax.text(0.1,3.55,"R-CNN-style pipeline",fontsize=10,color=BLUE,weight="bold")
    labels=["Image + region\nproposals (~2,000)","CNN features\nfor each crop","Classification +\nbox refinement","NMS +\nfinal detections"]
    for i,t in enumerate(labels):
        box(ax,0.1+i*2.52,2.25,2.25,0.9,t,fs=8)
        if i<3: arrow(ax,(2.35+i*2.52,2.70),(2.62+i*2.52,2.70))
    ax.text(0.1,1.82,"YOLOv1: one full-image network evaluation",fontsize=10,color=TEAL,weight="bold")
    for i,t in enumerate(["Resize whole image\n448 × 448","24 conv + 2 FC\nsingle network","Decode boxes +\nclass-specific scores","NMS +\nfinal detections"]):
        box(ax,0.1+i*2.52,0.55,2.25,0.9,t,color=TEAL,fs=8)
        if i<3: arrow(ax,(2.35+i*2.52,1),(2.62+i*2.52,1),TEAL)
    save(fig,"pipeline_comparison")


def grid():
    fig, ax = canvas(3.5)
    x0,y0,size=0.2,0.35,4.4
    ax.add_patch(Rectangle((x0,y0),size,size,facecolor="#FAFCFE",edgecolor=BLUE,lw=1))
    row,col=3,2
    # Origin at lower left; indexing in the labels remains top-to-bottom.
    gy=y0+(6-row)*size/7
    gx=x0+col*size/7
    ax.add_patch(Rectangle((gx,gy),size/7,size/7,facecolor="#DBEAF6",edgecolor="none"))
    for k in range(1,7):
        ax.plot([x0+k*size/7]*2,[y0,y0+size],color="#ADBFD0",lw=.6)
        ax.plot([x0,x0+size],[y0+k*size/7]*2,color="#ADBFD0",lw=.6)
    cx=x0+2.45/7*size
    cy=y0+(1-3.45/7)*size
    ax.add_patch(Rectangle((cx-.30*size/2,cy-.35*size/2),.30*size,.35*size,
                           facecolor="#237F8812",edgecolor=TEAL,lw=1.8))
    ax.plot(cx,cy,"o",color=ORANGE,ms=5)
    ax.text(x0,y0+size+.18,"7 × 7 spatial responsibility grid",fontsize=9,weight="bold",color=BLUE)
    ax.text(x0,y0-.22,"Box may extend beyond its responsible cell.",fontsize=8,color=INK)
    box(ax,5.3,3.45,4.4,1.0,"One cell: 30 output values\n2 × (x, y, w, h, confidence) + 20 classes",fs=8)
    box(ax,5.3,1.98,4.4,1.0,"Two geometric box predictors\nOne shared conditional class vector",color=TEAL,fs=9)
    box(ax,5.3,.5,4.4,1.0,"Whole image: 7 × 7 × 30\n98 box slots; 20 scores per slot",fs=9)
    arrow(ax,(5.0,3.95),(5.3,3.95))
    arrow(ax,(7.5,3.45),(7.5,2.98))
    arrow(ax,(7.5,1.98),(7.5,1.5))
    save(fig,"grid_representation")


def architecture():
    fig,ax=canvas(4.45)
    box(ax,2.7,5.75,4.6,.55,"RGB input: 448 × 448 × 3",fs=9)
    groups=[
        (0.1,4.1,"Stage 1 · 1 conv\n7×7 / 64, stride 2\n2×2 max pool, stride 2\n112 × 112 × 64"),
        (3.48,4.1,"Stage 2 · 1 conv\n3×3 / 192\n2×2 max pool, stride 2\n56 × 56 × 192"),
        (6.85,4.1,"Stage 3 · 4 conv\n1×1/128 → 3×3/256\n1×1/256 → 3×3/512; pool\n28 × 28 × 512"),
        (6.85,2.1,"Stage 4 · 10 conv\n[1×1/256 → 3×3/512] ×4\n1×1/512 → 3×3/1024; pool\n14 × 14 × 1024"),
        (3.48,2.1,"Stage 5 · 6 conv\n[1×1/512 → 3×3/1024] ×2\n3×3/1024 → 3×3/1024, s2\n7 × 7 × 1024"),
        (0.1,2.1,"Stage 6 · 2 conv\n3×3/1024 → 3×3/1024\nFinal convolutional features\n7 × 7 × 1024"),
    ]
    for x,y,t in groups:box(ax,x,y,3.03,1.32,t,fs=7.8)
    arrow(ax,(2.7,6.03),(1.62,5.42))
    arrow(ax,(3.13,4.76),(3.48,4.76));arrow(ax,(6.51,4.76),(6.85,4.76))
    arrow(ax,(8.37,4.1),(8.37,3.42))
    arrow(ax,(6.85,2.76),(6.51,2.76));arrow(ax,(3.48,2.76),(3.13,2.76))
    box(ax,.1,.6,3.03,.85,"Flatten → FC 4096\nLeaky ReLU + dropout (0.5)",color=TEAL,fs=8)
    box(ax,3.48,.6,3.03,.85,"FC 1470\nLinear prediction layer",color=TEAL,fs=9)
    box(ax,6.85,.6,3.03,.85,"Reshape 7 × 7 × 30\nBoxes + conf + class probs",color=TEAL,fs=7.8)
    arrow(ax,(1.62,2.1),(1.62,1.45));arrow(ax,(3.13,1.03),(3.48,1.03),TEAL)
    arrow(ax,(6.51,1.03),(6.85,1.03),TEAL)
    ax.text(5,.12,"Convolution count: 1 + 1 + 4 + 10 + 6 + 2 = 24; two fully connected layers.",
            ha="center",fontsize=8,color=INK)
    save(fig,"yolov1_architecture")


def loss_flow():
    fig,ax=canvas(3.0)
    box(ax,2.7,3.68,4.6,.65,"Training targets + current predictions",fs=8.5)
    box(ax,.12,2.18,4.6,.95,"Cell contains an object\nSelect its maximum-IoU predictor",color=TEAL,fs=9)
    box(ax,5.26,2.18,4.6,.95,"Cell contains no object\nAll box confidences target zero",fs=9)
    arrow(ax,(4,3.68),(2.42,3.13));arrow(ax,(6,3.68),(7.56,3.13))
    box(ax,.12,.66,4.6,.92,"Responsible box: coordinates ×5, conf ×1\nCell: shared class-probability SSE ×1",color=TEAL,fs=7.7)
    box(ax,5.26,.66,4.6,.92,"Confidence SSE ×0.5\nNo coordinate or classification loss",fs=8.5)
    arrow(ax,(2.42,2.18),(2.42,1.58),TEAL);arrow(ax,(7.56,2.18),(7.56,1.58))
    ax.text(5,.18,"Unassigned boxes use no-object confidence loss; only the responsible box learns coordinates.",
            fontsize=8,ha="center",color=INK)
    save(fig,"loss_assignment")


def inference():
    fig,ax=canvas(2.65)
    labels=["Input image\nresize: 448 × 448","One CNN forward\n7 × 7 × 30 tensor","Decode 98 boxes\nscore = class × conf"]
    for i,t in enumerate(labels):
        box(ax,.15+i*3.38,2.2,2.92,.9,t,fs=9)
        if i<2:arrow(ax,(3.07+i*3.38,2.65),(3.53+i*3.38,2.65))
    for x,t in [(6.91,"Score threshold\nretain candidate classes"),(3.53,"Class-wise NMS\nremove class duplicates"),(.15,"Final detections\nbox, label, score")]:
        box(ax,x,.6,2.92,.9,t,color=TEAL,fs=8.5)
    arrow(ax,(8.37,2.2),(8.37,1.5))
    arrow(ax,(6.91,1.05),(6.45,1.05),TEAL);arrow(ax,(3.53,1.05),(3.07,1.05),TEAL)
    ax.text(5,.12,"The confidence target contains IoU during training; ground truth is not used at inference.",
            ha="center",fontsize=8,color=INK)
    save(fig,"inference_pipeline")


def evolution():
    fig,ax=canvas(3.3)
    steps=[("v1","Unified\nprediction"),("v2–v3","Anchors +\nmulti-scale"),("v4–v7","Training +\ndeployment"),("v8–v10","Head + gradient\n+ NMS-free"),("11–v12","Selective to\ncentral attention")]
    for i,(name,text) in enumerate(steps):
        x=.1+i*2.02
        box(ax,x,2.8,1.76,1.02,f"{name}\n{text}",fs=8)
        if i<4:arrow(ax,(x+1.76,3.31),(x+2.02,3.31))
    box(ax,.1,.62,4.72,1.28,"Parallel research branches\nUltralytics: v5 → v8 → YOLO11\nWang et al.: v7 → v9",color=TEAL,fs=9)
    box(ax,5.17,.62,4.72,1.28,"Independent optimization questions\nv10: duplicate predictions and NMS\nv12: efficient attention and aggregation",fs=9)
    ax.text(5,.13,"Arrows summarize technical themes, not a single codebase or strict inheritance chain.",
            ha="center",fontsize=8,color=INK)
    save(fig,"evolution_overview")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for function in [pipelines,grid,loss_flow,inference,evolution]:
        function()
    print("Generated five active schematics; architecture illustration remains reserved in the report.")
