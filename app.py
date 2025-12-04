import streamlit as st
from transformers import pipeline
import time

# Set page config
st.set_page_config(
    page_title="AI Text Generator",
    page_icon="✨",
    layout="wide"
)

# Title
st.title("✨ AI Text Generator with Sentiment")
st.markdown("Generate paragraphs matching your prompt's sentiment")

# Cache models to avoid reloading
@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    """Load only the sentiment model"""
    return pipeline("sentiment-analysis", 
                   model="distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_resource(show_spinner=False)
def load_generation_model():
    """Load text generation model"""
    return pipeline("text-generation", 
                   model="gpt2",
                   max_length=200)

# Load models with progress
with st.spinner("Loading AI models (first time may take a minute)..."):
    try:
        sentiment_analyzer = load_sentiment_model()
        text_generator = load_generation_model()
        st.success("Models loaded successfully!")
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Please check your internet connection and try again.")
        st.stop()

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Sentiment options
    sentiment_mode = st.radio(
        "Sentiment:",
        ["Auto Detect", "Manual Select"]
    )
    
    if sentiment_mode == "Manual Select":
        selected_sentiment = st.selectbox(
            "Choose sentiment:",
            ["😊 Positive", "😠 Negative", "😐 Neutral"]
        )
    else:
        selected_sentiment = None
    
    # Length options
    output_length = st.slider(
        "Output length (words approx):",
        min_value=50,
        max_value=300,
        value=150,
        step=50
    )
    
    # Creativity
    temperature = st.slider(
        "Creativity:",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Lower = more focused, Higher = more creative"
    )
    
    st.markdown("---")
    st.markdown("**How to use:**")
    st.markdown("""
    1. Enter any text prompt
    2. Choose sentiment (auto or manual)
    3. Adjust length & creativity
    4. Click Generate!
    """)

# Main interface
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📝 Enter Your Prompt")
    prompt = st.text_area(
        "What would you like to write about?",
        placeholder="Example: 'The beauty of nature in springtime' or 'The challenges of modern city life'...",
        height=120,
        label_visibility="collapsed"
    )

with col2:
    st.subheader("💡 Examples")
    examples = st.selectbox(
        "Try these examples:",
        [
            "Select an example...",
            "The joy of learning something new",
            "The frustration of traffic jams",
            "The importance of renewable energy",
            "Memories of childhood summers",
            "The impact of social media on society"
        ]
    )
    
    if examples != "Select an example...":
        prompt = examples

# Generate button
generate_btn = st.button("🚀 Generate Text", type="primary", use_container_width=True)

# Results section
if generate_btn:
    if not prompt or prompt.strip() == "":
        st.warning("Please enter a prompt first!")
    else:
        with st.spinner("Analyzing and generating..."):
            
            # Step 1: Determine sentiment
            if sentiment_mode == "Auto Detect":
                try:
                    # Analyze sentiment
                    sentiment_result = sentiment_analyzer(prompt[:512])[0]
                    label = sentiment_result['label']
                    confidence = sentiment_result['score']
                    
                    # Map to our format
                    sentiment_map = {
                        "POSITIVE": "😊 Positive",
                        "NEGATIVE": "😠 Negative",
                        "LABEL_0": "😐 Neutral",  # Some models use different labels
                        "LABEL_1": "😊 Positive"
                    }
                    
                    detected_sentiment = sentiment_map.get(label, "😐 Neutral")
                    
                    # Display sentiment
                    sentiment_col1, sentiment_col2 = st.columns(2)
                    with sentiment_col1:
                        st.metric("Detected Sentiment", detected_sentiment)
                    with sentiment_col2:
                        st.metric("Confidence", f"{confidence:.1%}")
                    
                    final_sentiment = detected_sentiment
                    
                except Exception as e:
                    st.warning(f"Could not detect sentiment: {str(e)}. Using neutral.")
                    final_sentiment = "😐 Neutral"
            else:
                final_sentiment = selected_sentiment
            
            # Step 2: Prepare generation prompt based on sentiment
            sentiment_prefixes = {
                "😊 Positive": "Write a positive and optimistic text about: ",
                "😠 Negative": "Write a critical or negative perspective on: ",
                "😐 Neutral": "Write a balanced and objective text about: ",
                "Positive": "Write a positive and optimistic text about: ",
                "Negative": "Write a critical or negative perspective on: ",
                "Neutral": "Write a balanced and objective text about: "
            }
            
            prefix = sentiment_prefixes.get(final_sentiment, "Write about: ")
            guided_prompt = prefix + prompt
            
            # Step 3: Generate text
            try:
                # Calculate max length based on word count
                max_length = min(output_length * 2, 500)  # Rough conversion
                
                generated = text_generator(
                    guided_prompt,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    num_return_sequences=1
                )
                
                generated_text = generated[0]['generated_text']
                
                # Clean up: remove the guiding prefix if present
                if guided_prompt in generated_text:
                    generated_text = generated_text.replace(guided_prompt, "").strip()
                
                # Display results
                st.markdown("---")
                st.subheader("📄 Generated Text")
                
                # Text box with copy option
                st.text_area(
                    "Your generated text:",
                    generated_text,
                    height=300,
                    label_visibility="collapsed"
                )
                
                # Stats
                word_count = len(generated_text.split())
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Words", word_count)
                with col2:
                    st.metric("Sentiment", final_sentiment)
                with col3:
                    st.metric("Creativity", f"{temperature:.1f}")
                
                # Copy button
                if st.button("📋 Copy to Clipboard"):
                    st.code(generated_text)
                    st.success("Text copied! (You can also select and Ctrl+C)")
                
            except Exception as e:
                st.error(f"Error generating text: {str(e)}")
                st.info("Try reducing the length or using a simpler prompt.")

# Footer
st.markdown("---")
with st.expander("ℹ️ About this app"):
    st.markdown("""
    ### AI Text Generator with Sentiment Control
    
    **How it works:**
    1. **Sentiment Analysis**: Uses DistilBERT model to detect if your prompt is Positive, Negative, or Neutral
    2. **Text Generation**: Uses GPT-2 to generate coherent paragraphs
    3. **Sentiment Guidance**: Guides the AI to match the detected/selected sentiment
    
    **Models used:**
    - 🤖 Sentiment: `distilbert-base-uncased-finetuned-sst-2-english`
    - ✍️ Generation: `gpt2`
    
    **Note**: First run downloads models (~500MB). Subsequent runs are faster!
    """)

st.caption("Built with Streamlit & HuggingFace Transformers | For AI Internship Assessment")
