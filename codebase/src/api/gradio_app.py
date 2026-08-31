"""
Gradio interactive demo for Root Cause Prediction.
Run: python -m src.api.gradio_app
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr
import yaml

from src.pipeline.predictor import RootCausePredictor
from src.utils.helpers import load_config


def format_result(result: dict) -> str:
    lines = [
        f"### Predicted Root Cause: **{result['predicted_root_cause']}**",
        f"**Confidence:** {result['confidence']:.2%}",
        "",
        "### Similar Historical Failures",
    ]
    for i, s in enumerate(result.get("similar_historical_failures", []), 1):
        lines.append(
            f"{i}. `{s['failure_id']}` | **{s['category']}** | sim={s['similarity']:.3f}  \n"
            f"   Service: {s.get('service', 'N/A')}  \n"
            f"   {s['error_message']}"
        )
    lines.append("")
    lines.append("### Cleaned Input Preview")
    lines.append(f"```\n{result.get('cleaned_input_preview', '')}\n```")
    return "\n".join(lines)


def build_demo(artifacts_dir: str = "artifacts", config_path: str = "config/config.yaml"):
    cfg = load_config(config_path)
    predictor = RootCausePredictor.load(artifacts_dir, config_path)

    def predict_fn(error_text: str, top_k: int):
        if not error_text or not error_text.strip():
            return "Please paste an error message or stacktrace."
        result = predictor.predict(error_text, top_k_similar=int(top_k))
        return format_result(result)

    example = """Service: payment-service
Error: java.lang.NullPointerException: Cannot invoke "com.example.User.getId()" because "user" is null
at com.example.PaymentService.process(PaymentService.java:112)
at org.springframework.web.servlet.DispatcherServlet.doDispatch(DispatcherServlet.java:1072)"""

    demo = gr.Interface(
        fn=predict_fn,
        inputs=[
            gr.Textbox(
                lines=12,
                label="Error / Stacktrace",
                placeholder="Paste full error message + stacktrace here...",
                value=example,
            ),
            gr.Slider(1, 10, value=5, step=1, label="Number of similar cases"),
        ],
        outputs=gr.Markdown(label="Prediction"),
        title=cfg.get("gradio", {}).get("title", "AI Failure Root Cause Predictor"),
        description=cfg.get("gradio", {}).get(
            "description",
            "Paste a stacktrace or error message to get the predicted root cause and similar historical failures.",
        ),
        examples=[
            [example, 5],
            [
                "Service: order-service\nError: OptimisticLockException: version mismatch on order entity\nat com.example.OrderService.update(OrderService.java:88)",
                3,
            ],
            [
                "Service: api-gateway\nError: SocketTimeoutException: Read timed out after 30000 ms\nat java.net.SocketInputStream.socketRead0",
                5,
            ],
        ],
        allow_flagging="never",
    )
    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(share=False, server_name="0.0.0.0")
