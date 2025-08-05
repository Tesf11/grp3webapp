import streamlit as st
import pandas as pd

def render_model1_ui(model_input, model_name, mapping_data=None, debug=False):
    """
    Renders the UI for ANS Import & Export Storage Bin Prediction.
    
    Parameters:
        model_input: Either a full model_package dict or a trained sklearn model.
        model_name: Name of the model (string).
        mapping_data: Optional dict containing 'category_to_product_type' and 'product_type_to_bin'
                      if not included in model_input (for .pkl + .json setup).
        debug: Bool to print debug output.
    """

    st.subheader("🔍 ANS Import & Export - Storage Bin Prediction")

    st.markdown("""
    **Problem Statement:**  
    ANS Import and Export faces challenges in locating samples because they are split between two locations: the **Office** and **Warehouse Bins**.  
    This model predicts the correct **storage bin** for a product sample based on its **text description** and **weight** to automate and speed up sorting.

    **Inputs:**  
    - **Text Description** (e.g., Sub-category or product name)  
    - **Weight** (grams)  

    **Outputs:**  
    - **Bin 1**: Bulk & Perishables *(food, rice, fruits, snacks, drinks)*  
    - **Bin 2**: Durables & Utilities *(tools, furniture, toys, appliances)*  
    - **Bin 3**: Sensitive & Spill-Risk *(liquids, cosmetics, toiletries)*  
    - **Office**: Controlled & Tracked *(electronics, office supplies, media, baby products)*
    """)

    # --- Determine if model_input is a package or just a model ---
    if isinstance(model_input, dict) and "model" in model_input:
        model = model_input["model"]
        category_to_product_type = {
            k.lower(): v for k, v in model_input["category_to_product_type"].items()
        }
        product_type_to_bin = {
            k.lower(): v for k, v in model_input["product_type_to_bin"].items()
        }
    else:
        model = model_input
        if mapping_data is None:
            st.error("Mapping data is required when using model-only format (.pkl + .json).")
            return
        category_to_product_type = {
            k.lower(): v for k, v in mapping_data.get("category_to_product_type", {}).items()
        }
        product_type_to_bin = {
            k.lower(): v for k, v in mapping_data.get("product_type_to_bin", {}).items()
        }

    # --- User Inputs ---
    description = st.text_input(f"[{model_name}] Enter product description", key=f"desc_{model_name}")
    weight_g = st.number_input(f"[{model_name}] Enter product weight (grams)", min_value=0.0, key=f"weight_{model_name}")

    if st.button(f"Predict with {model_name}"):
        if description and weight_g > 0:
            # Prepare input DataFrame
            input_df = pd.DataFrame([{"description": description, "weight_g": weight_g}])

            # Step 1: Predict category
            try:
                predicted_category = model.predict(input_df)[0]
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                return

            # Step 2: Map to product type (case-insensitive)
            product_type = category_to_product_type.get(str(predicted_category).lower())

            if not product_type:
                st.warning(f"Could not map predicted category '{predicted_category}' to a product type.")
                return

            # Step 3: Map to storage bin
            storage_bin = product_type_to_bin.get(str(product_type).lower(), "Unknown")

            # --- Output results ---
            st.success(f"**Predicted Category:** {predicted_category}")
            st.info(f"**Product Type:** {product_type}")
            st.success(f"**Assigned Storage Bin:** {storage_bin}")

            if debug:
                st.write("🔍 Debug Info:", {
                    "Predicted Category": predicted_category,
                    "Product Type": product_type,
                    "Storage Bin": storage_bin
                })

        else:
            st.warning("Please enter both a description and a valid weight.")
