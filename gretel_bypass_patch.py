"""
Patch for Gretel DataDesigner to bypass Gretel servers for custom model API calls.
This patch intercepts the preview() method and executes LLM tasks locally when
custom connections are detected.
"""

import json
import logging
import requests
from typing import Any, Dict, List, Optional

import pandas as pd
from gretel_client.data_designer.data_designer import DataDesigner
from gretel_client.data_designer.preview import PreviewResults
from gretel_client.data_designer.types import LLMTextColumn, LLMGenColumn
from gretel_client.workflows.configs.workflows import ModelConfig

logger = logging.getLogger(__name__)


class LocalLLMExecutor:
    """Handles local execution of LLM tasks for custom model connections."""

    def __init__(self, connection_configs: Dict[str, Dict[str, Any]]):
        self.connection_configs = connection_configs

    def execute_llm_column(
        self, column, context_data: Dict[str, Any], model_config: ModelConfig
    ) -> str:
        """Execute an LLM column generation task locally."""
        if not model_config.connection_id:
            raise ValueError("Model config must have connection_id for local execution")

        connection_config = self.connection_configs.get(model_config.connection_id)
        if not connection_config:
            raise ValueError(
                f"Connection config not found for ID: {model_config.connection_id}"
            )

        api_base = connection_config.get("api_base")
        api_key = connection_config.get("api_key", "local-key")

        if not api_base:
            raise ValueError("api_base not found in connection config")

        # Prepare the prompt by substituting template variables
        prompt = self._substitute_template_variables(column.prompt, context_data)

        # Make direct API call to local LLM endpoint
        return self._call_local_llm(
            api_base=api_base,
            api_key=api_key,
            model_name=model_config.model_name,
            system_prompt=getattr(column, "system_prompt", ""),
            prompt=prompt,
            generation_params=model_config.generation_parameters,
        )

    def _substitute_template_variables(
        self, prompt: str, context_data: Dict[str, Any]
    ) -> str:
        """Substitute template variables in the prompt with actual values."""
        import re

        # Find all template variables like {{variable_name}}
        template_vars = re.findall(r"\{\{(\w+)\}\}", prompt)

        for var in template_vars:
            if var in context_data:
                prompt = prompt.replace(f"{{{{{var}}}}}", str(context_data[var]))

        return prompt

    def _call_local_llm(
        self,
        api_base: str,
        api_key: str,
        model_name: str,
        system_prompt: str,
        prompt: str,
        generation_params: Any,
    ) -> str:
        """Make a direct API call to the local LLM endpoint."""
        url = f"{api_base.rstrip('/')}/chat/completions"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Handle GenerationParameters object and nested objects
        if hasattr(generation_params, "__dict__"):
            params_dict = {}
            for k, v in generation_params.__dict__.items():
                if v is not None:
                    # Handle nested objects like MaxTokens
                    if hasattr(v, "__dict__"):
                        # Extract the actual value from objects like MaxTokens
                        if hasattr(v, "value"):
                            params_dict[k] = v.value
                        elif hasattr(v, "__dict__") and len(v.__dict__) == 1:
                            # Single attribute object, use its value
                            params_dict[k] = next(iter(v.__dict__.values()))
                        else:
                            # Convert to dict if complex object
                            params_dict[k] = str(v)
                    else:
                        params_dict[k] = v
        elif isinstance(generation_params, dict):
            params_dict = generation_params
        else:
            params_dict = {}

        payload = {"model": model_name, "messages": messages, **params_dict}

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        logger.info(f"🔄 Making local LLM call to {url}")
        logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()

            result = response.json()
            content = (
                result.get("choices", [{}])[0].get("message", {}).get("content", "")
            )

            if not content:
                raise ValueError("Empty response from local LLM")

            logger.info(f"✅ Local LLM response: {content[:100]}...")
            return content

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Local LLM call failed: {e}")
            raise


def patch_data_designer():
    """Apply the bypass patch to DataDesigner."""

    def _has_custom_connections(self) -> bool:
        """Check if the DataDesigner has custom model connections."""
        if not self._model_configs:
            return False

        for model_config in self._model_configs:
            if model_config.connection_id:
                return True
        return False

    def _get_connection_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get connection configurations for custom models."""
        # In a real implementation, this would fetch from the connections API
        # For now, we'll use a simple mapping approach
        connection_configs = {}

        # This is a simplified approach - in practice you'd need to fetch
        # the actual connection configs from Gretel's API
        for model_config in self._model_configs or []:
            if model_config.connection_id:
                # For this demo, we'll assume the connection config is available
                # In practice, you'd fetch this from the connections API
                connection_configs[model_config.connection_id] = {
                    "api_base": "http://127.0.0.1:1234/v1",  # Default local LM Studio
                    "api_key": "local-key",
                }

        return connection_configs

    def _execute_local_preview(self, verbose_logging: bool = False) -> PreviewResults:
        """Execute preview locally without going through Gretel servers."""
        logger.info("🚀 Executing local preview (bypassing Gretel servers)")

        # Get connection configurations
        connection_configs = self._get_connection_configs()
        executor = LocalLLMExecutor(connection_configs)

        # Generate sample data
        num_records = 10  # Preview records
        records = []

        try:
            for i in range(num_records):
                record = {}

                # Process columns in dependency order
                sorted_columns = self._get_sorted_columns()

                for column_name in sorted_columns:
                    column = self.get_column(column_name)

                    if hasattr(column, "params"):
                        # Handle different sampler types
                        import random
                        
                        if hasattr(column.params, "values"):
                            if isinstance(column.params.values, list):
                                # Category sampler - simple list of values
                                record[column_name] = random.choice(column.params.values)
                            elif isinstance(column.params.values, dict):
                                # Subcategory sampler - depends on category column
                                category_name = getattr(column.params, "category", None)
                                if category_name and category_name in record:
                                    category_value = record[category_name]
                                    subcategories = column.params.values.get(category_value, [])
                                    if subcategories:
                                        record[column_name] = random.choice(subcategories)
                                    else:
                                        record[column_name] = f"[No subcategories for {category_value}]"
                                else:
                                    record[column_name] = f"[Missing category dependency: {category_name}]"
                            else:
                                record[column_name] = str(column.params.values)

                    elif isinstance(column, (LLMTextColumn, LLMGenColumn)):
                        # LLM column - execute locally
                        model_config = self._get_model_config_for_column(column)
                        if model_config and model_config.connection_id:
                            try:
                                result = executor.execute_llm_column(
                                    column, record, model_config
                                )
                                record[column_name] = result
                            except Exception as e:
                                logger.error(
                                    f"❌ Failed to execute LLM column {column_name}: {e}"
                                )
                                record[column_name] = f"[ERROR: {str(e)}]"
                        else:
                            record[column_name] = (
                                f"[No custom connection for {column_name}]"
                            )

                    else:
                        # Other column types - basic handling
                        record[column_name] = f"sample_{column_name}_{i}"

                records.append(record)
                logger.info(f"✅ Generated record {i+1}/{num_records}")

            # Create DataFrame
            df = pd.DataFrame(records)

            # Create PreviewResults - use correct import path
            from gretel_client.data_designer.viz_tools import AIDDMetadata

            # Create minimal AIDDMetadata for preview
            aidd_metadata = AIDDMetadata(
                columns=list(df.columns),
                num_records=len(df),
                project_id="local-preview",
            )

            preview_results = PreviewResults(
                aidd_metadata=aidd_metadata,
                output=df,
                success=True,
                evaluation_results={},
            )

            logger.info("🎉 Local preview completed successfully!")
            return preview_results

        except Exception as e:
            logger.error(f"❌ Local preview failed: {e}")
            # Return empty results on failure
            from gretel_client.data_designer.viz_tools import AIDDMetadata

            # Create minimal AIDDMetadata for error case
            aidd_metadata = AIDDMetadata(
                columns=[], num_records=0, project_id="local-preview-error"
            )

            return PreviewResults(
                aidd_metadata=aidd_metadata,
                output=None,
                success=False,
                evaluation_results={"error": f"Local execution failed: {str(e)}"},
            )

    def _get_sorted_columns(self) -> List[str]:
        """Get columns sorted by dependencies."""
        # Separate column types and handle dependencies
        category_columns = []
        subcategory_columns = []
        llm_columns = []
        other_columns = []

        for column_name, column in self._columns.items():
            if hasattr(column, "params"):
                if hasattr(column.params, "values"):
                    if isinstance(column.params.values, list):
                        # Category sampler - no dependencies
                        category_columns.append(column_name)
                    elif isinstance(column.params.values, dict):
                        # Subcategory sampler - depends on category
                        subcategory_columns.append(column_name)
                    else:
                        other_columns.append(column_name)
                else:
                    other_columns.append(column_name)
            elif hasattr(column, "model_alias"):
                # LLM columns - should come after samplers
                llm_columns.append(column_name)
            else:
                other_columns.append(column_name)

        # Return in dependency order: categories first, then subcategories, then LLM, then others
        return category_columns + subcategory_columns + llm_columns + other_columns

    def _get_model_config_for_column(self, column) -> Optional[ModelConfig]:
        """Get the model config for a specific column."""
        if not hasattr(column, "model_alias"):
            return None

        for model_config in self._model_configs or []:
            if model_config.alias == column.model_alias:
                return model_config

        return None

    def patched_preview(
        self, verbose_logging: bool = False, validate: bool = True
    ) -> PreviewResults:
        """Patched preview method that bypasses Gretel servers for custom connections."""
        if validate:
            self._run_semantic_validation(raise_exceptions=True)

        # Check if this configuration uses custom model connections
        if self._has_custom_connections():
            logger.info("🔄 Detected custom model connections - executing locally")
            return self._execute_local_preview(verbose_logging=verbose_logging)
        else:
            # Use original implementation for non-custom connections
            logger.info("🚀 Generating preview (using Gretel servers)")
            workflow = self._build_workflow(
                verbose_logging=verbose_logging, streaming=True
            )
            return self._capture_preview_result(
                workflow, verbose_logging=verbose_logging
            )

    # Apply the patch
    DataDesigner._has_custom_connections = _has_custom_connections
    DataDesigner._get_connection_configs = _get_connection_configs
    DataDesigner._execute_local_preview = _execute_local_preview
    DataDesigner._get_sorted_columns = _get_sorted_columns
    DataDesigner._get_model_config_for_column = _get_model_config_for_column
    DataDesigner.preview = patched_preview

    logger.info("✅ DataDesigner bypass patch applied successfully!")


def apply_patch():
    """Apply the Gretel bypass patch."""
    patch_data_designer()
    print("🔧 Gretel DataDesigner bypass patch applied!")
    print("   Custom model API calls will now bypass Gretel servers")
    print("   and route directly to your local endpoints.")


if __name__ == "__main__":
    apply_patch()
