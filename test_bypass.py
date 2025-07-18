"""
Test script to demonstrate the Gretel bypass functionality.
This script shows how custom model API calls bypass Gretel servers.
"""

import logging
import sys
from gretel_bypass_patch import apply_patch

# Apply the bypass patch
apply_patch()

# Now import Gretel components after patch is applied
from gretel_client.navigator_client import Gretel
from gretel_client.workflows.configs.workflows import ModelConfig
from gretel_client.data_designer import columns as C
from gretel_client.data_designer import params as P

# Enable logging to see the bypass in action
logging.basicConfig(level=logging.INFO)


def test_bypass_functionality():
    """Test that custom model connections bypass Gretel servers."""

    print("🧪 Testing Gretel DataDesigner bypass functionality...")

    # Initialize Gretel client
    gretel = Gretel(
        api_key="grtua4cc39c0083296fb064e347552491b06e3cb798d5d7745d9f94e1e7d09f35414"
    )

    # Create API connection to local LM Studio
    print("📡 Creating local LLM connection...")
    api_base_url = "http://127.0.0.1:1234/v1"

    try:
        connection_id = gretel.data_designer.create_api_key_connection(
            name="test-local-llm",
            api_base=api_base_url,
            api_key="local-key",
        )
        print(f"✅ Created connection with ID: {connection_id}")

        # Create Data Designer with custom model suite
        print("🏗️ Creating Data Designer with custom model...")
        data_designer = gretel.data_designer.new(
            model_suite="bring-your-own",
            model_configs=[
                ModelConfig(
                    alias="local-llm",
                    model_name="qwen2.5-7b-instruct",
                    connection_id=connection_id,
                    generation_parameters={"temperature": 0.7, "max_tokens": 100},
                )
            ],
        )

        # Add columns
        print("📊 Adding columns...")
        data_designer.add_column(
            C.SamplerColumn(
                name="product_category",
                type=P.SamplerType.CATEGORY,
                params=P.CategorySamplerParams(
                    values=["Electronics", "Books", "Clothing"]
                ),
            )
        )
        data_designer.add_column(
            C.SamplerColumn(
                name="product_subcategory",
                type=P.SamplerType.SUBCATEGORY,
                params=P.SubcategorySamplerParams(
                    category="product_category",
                    values={
                        "Electronics": ["Smartphones", "Laptops", "Headphones"],
                        "Books": ["Fiction", "Non-Fiction", "Biography"],
                        "Clothing": ["Shirts", "Pants", "Dresses", "Shoes"],
                    },
                ),
            )
        )

        data_designer.add_column(
            C.LLMTextColumn(
                name="product_name",
                system_prompt="You are a product name generator.",
                prompt="Generate a short product name for the {{product_category}} category within the subcategory: {{product_subcategory}}.",
                model_alias="local-llm",
            )
        )

        data_designer.add_column(
            C.LLMTextColumn(
                name="product_description",
                system_prompt="You are a product description generator.",
                prompt="Generate a short product description for the {{product_category}} category and this is the product name: {{product_name}}.",
                model_alias="local-llm",
            )
        )

        print("🔍 Starting preview - this should bypass Gretel servers...")

        # This should now bypass Gretel servers and call local LLM directly
        preview = data_designer.preview()

        if preview.success:
            print("🎉 Success! Custom model API calls bypassed Gretel servers!")
            print(f"📊 Generated {len(preview.output)} preview records")
            print("\nSample data:")
            # print(preview.output.head())
            print(preview.dataset.df.head(10))
            # save to csv
            preview.output.to_csv("preview.csv", index=False)
        else:
            print("❌ Preview failed")
            print(f"Logs: {preview.evaluation_results}")

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    return True


if __name__ == "__main__":
    print("🔧 Gretel DataDesigner Bypass Test")
    print("=" * 50)

    success = test_bypass_functionality()

    if success:
        print("\n✅ Test completed successfully!")
        print(
            "🚀 The bypass patch is working - custom model calls now go directly to local endpoints!"
        )
    else:
        print("\n❌ Test failed!")
        print("Check your local LLM server and connection settings.")

    sys.exit(0 if success else 1)
