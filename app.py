
import joblib
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="OpsLens | Ticket Routing",
    page_icon="🎫",
    layout="wide",
)

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "opslens_ticket_classifier.joblib"
)

TEAMS = {
    "application": "Application support",
    "network": "Network support",
    "database": "Database support",
    "access": "Identity and access support",
}

EXAMPLES = {
    "VPN drops": "My VPN connection drops every time I join a video call.",
    "Sign-in blocked": "My password reset link expired and I cannot sign in.",
    "Portal freezes": "The web app freezes when I submit a form.",
    "Query blocked": "The database query is waiting for a locked table.",
}


@st.cache_resource
def load_model(path, modified_ns):
    artifact = joblib.load(path)
    # Supports both the standalone pipeline and the advanced notebook bundle.
    return artifact["pipeline"] if isinstance(artifact, dict) else artifact


def clear_prediction():
    st.session_state.pop("category", None)


def use_example(text):
    st.session_state.ticket = text
    clear_prediction()


st.html("""
<style>
  .stApp { background: #FAF8F2; }
  .ops-kicker { color: #176350; font-weight: 700; letter-spacing: .08em; }
  .ops-heading h1 { color: #132E28; font-size: clamp(2rem, 5vw, 4rem); }
  .ops-heading p { color: #415D55; }
  .st-key-result {
    background: #E8F2E8;
    border: 1px solid #C8DBCE;
    border-radius: 1rem;
    padding: 1.25rem;
  }
</style>

<div class="ops-heading">
  <div class="ops-kicker">OPSLENS / IT SUPPORT</div>
  <h1>Which team should handle this ticket?</h1>
  <p>Describe the issue to get a suggested support category.</p>
</div>
""")

if not MODEL_PATH.is_file():
    st.error("The model file is missing. Run the model export cell in Colab.")
    st.stop()

entry, result = st.columns([3, 2], gap="large")

with entry:
    st.subheader("Ticket details")
    st.caption("Choose an example or enter your own fictional ticket.")

    for column, (label, text) in zip(st.columns(4), EXAMPLES.items()):
        column.button(label, on_click=use_example, args=(text,))

    description = st.text_area(
        "Ticket description",
        key="ticket",
        height=150,
        max_chars=2000,
        placeholder="What happened, and which system was affected?",
        on_change=clear_prediction,
    )

    if st.button("Suggest category", type="primary"):
        clear_prediction()
        if not description.strip():
            st.warning("Enter a ticket description first.")
        else:
            model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
            prediction = model.predict([description.strip()])[0]
            st.session_state.category = str(prediction).lower()

with result:
    with st.container(key="result"):
        st.subheader("Routing suggestion")
        category = st.session_state.get("category")

        if category:
            st.success(category.title())
            st.write(f"Suggested team: **{TEAMS[category]}**")
            st.caption("Review the suggestion before assigning the ticket.")
        else:
            st.write("Your suggested category will appear here.")

st.caption("Linkzie Technologies · OpsLens workshop demo · Synthetic training data")
