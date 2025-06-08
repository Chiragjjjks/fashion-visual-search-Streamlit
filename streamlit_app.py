import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from ast import literal_eval
from app.search_engine import search_similar

st.set_page_config(page_title="Fashion Visual Search", layout="centered")

st.title("🧠 Fashion Visual Search Assistant")
st.markdown("Paste an image URL to find visually similar fashion products.")

image_url = st.text_input("Enter Image URL:")

if image_url:
    try:
        response = requests.get(image_url, timeout=10)
        query_img = Image.open(BytesIO(response.content)).convert("RGB")
        st.image(query_img, caption="Query Image", use_column_width=True)

        with st.spinner("Searching for similar products..."):
            results = search_similar(image_url, top_k=5)

        st.markdown("### 🔎 Similar Products")
        if results is not None:
            for _, row in results.iterrows():
                st.markdown(
                    f"**Category:** {row['category_type']} &nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp; "
                    f"**Distance:** `{row['distance']:.4f}`"
                )

                try:
                    img_urls = literal_eval(row["pdp_images_s3"]) if isinstance(row["pdp_images_s3"], str) else []
                    if img_urls:
                        cols = st.columns(len(img_urls))
                        for col, img_url in zip(cols, img_urls):
                            with col:
                                st.image(img_url, width=150)
                except Exception as e:
                    st.error(f"⚠️ Error loading image list: {e}")
        else:
            st.error("❌ Could not fetch results. Please check the image URL.")

    except Exception as e:
        st.error(f"❌ Error loading query image: {e}")
