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

    def __init__(self, connection_configs: Dict[str, Dict[str, Any]], debug_storage: bool = False, debug_file: Optional[str] = None):
        self.connection_configs = connection_configs
        self.debug_storage = debug_storage
        self.debug_file = debug_file
        self.debug_calls = [] if debug_storage else None
        self.debug_call_count = 0
        
        # Initialize debug file if streaming is enabled
        if debug_storage and debug_file:
            self._initialize_debug_file()

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

        # Store debug information if enabled
        debug_info = None
        if self.debug_storage:
            debug_info = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "url": url,
                "payload": payload.copy(),
                "headers": {k: v for k, v in headers.items() if k != "Authorization"},
                "model_name": model_name,
                "system_prompt": system_prompt,
                "user_prompt": prompt,
            }

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
            
            # Store debug information if enabled
            if self.debug_storage and debug_info:
                debug_info.update({
                    "success": True,
                    "response": result,
                    "response_content": content,
                    "response_length": len(content),
                })
                
                # Append to memory if not streaming to file
                if self.debug_calls is not None:
                    self.debug_calls.append(debug_info)
                
                # Append to file in real-time if streaming enabled
                if self.debug_file:
                    self._append_debug_call(debug_info)
            
            return content

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Local LLM call failed: {e}")
            
            # Store debug information if enabled
            if self.debug_storage and debug_info:
                debug_info.update({
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                })
                
                # Append to memory if not streaming to file
                if self.debug_calls is not None:
                    self.debug_calls.append(debug_info)
                
                # Append to file in real-time if streaming enabled
                if self.debug_file:
                    self._append_debug_call(debug_info)
            
            raise

    def _initialize_debug_file(self):
        """Initialize the debug file with an empty JSON array."""
        try:
            with open(self.debug_file, "w") as f:
                f.write("[\n")
            logger.info(f"🔧 Initialized debug file: {self.debug_file}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize debug file {self.debug_file}: {e}")

    def _append_debug_call(self, debug_info: Dict[str, Any]):
        """Append a single debug call to the file in real-time."""
        if not self.debug_file:
            return
            
        try:
            # Add comma if not the first entry
            prefix = ",\n" if self.debug_call_count > 0 else ""
            
            with open(self.debug_file, "a") as f:
                f.write(f"{prefix}  {json.dumps(debug_info, indent=2, default=str)}")
                f.flush()  # Ensure immediate write
                
            self.debug_call_count += 1
            logger.debug(f"📝 Appended debug call #{self.debug_call_count} to {self.debug_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to append debug call to {self.debug_file}: {e}")

    def _finalize_debug_file(self):
        """Close the JSON array in the debug file."""
        if not self.debug_file:
            return
            
        try:
            with open(self.debug_file, "a") as f:
                f.write("\n]")
            logger.info(f"✅ Finalized debug file: {self.debug_file} ({self.debug_call_count} calls)")
        except Exception as e:
            logger.error(f"❌ Failed to finalize debug file {self.debug_file}: {e}")

    def save_debug_calls(self, filename: str = "llm_debug_calls.json"):
        """Save debug calls to a JSON file."""
        if not self.debug_storage or not self.debug_calls:
            logger.warning("No debug calls to save (debug_storage disabled or no calls made)")
            return
        
        with open(filename, "w") as f:
            json.dump(self.debug_calls, f, indent=2, default=str)
        
        logger.info(f"💾 Saved {len(self.debug_calls)} debug calls to {filename}")

    def get_debug_summary(self) -> Dict[str, Any]:
        """Get a summary of debug calls."""
        if not self.debug_storage or not self.debug_calls:
            return {"total_calls": 0, "successful_calls": 0, "failed_calls": 0}
        
        successful = sum(1 for call in self.debug_calls if call.get("success", False))
        failed = len(self.debug_calls) - successful
        
        return {
            "total_calls": len(self.debug_calls),
            "successful_calls": successful,
            "failed_calls": failed,
            "average_response_length": sum(call.get("response_length", 0) for call in self.debug_calls if call.get("success", False)) / max(successful, 1),
            "calls": self.debug_calls,
        }


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

    def _execute_local_preview(self, verbose_logging: bool = False, debug_storage: bool = False, debug_file: Optional[str] = None) -> PreviewResults:
        """Execute preview locally without going through Gretel servers."""
        logger.info("🚀 Executing local preview (bypassing Gretel servers)")

        # Get connection configurations
        connection_configs = self._get_connection_configs()
        executor = LocalLLMExecutor(connection_configs, debug_storage=debug_storage, debug_file=debug_file)
        
        # Store executor for later access to debug info
        self._last_executor = executor

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
            
            # Finalize debug file if streaming
            if debug_file and executor.debug_file:
                executor._finalize_debug_file()
            
            # Add debug info to results if enabled
            if debug_storage and executor.debug_storage:
                preview_results.debug_info = executor.get_debug_summary()
                logger.info(f"🔍 Debug info: {preview_results.debug_info['total_calls']} LLM calls captured")
            
            return preview_results

        except Exception as e:
            logger.error(f"❌ Local preview failed: {e}")
            # Return empty results on failure
            from gretel_client.data_designer.viz_tools import AIDDMetadata

            # Create minimal AIDDMetadata for error case
            aidd_metadata = AIDDMetadata(
                columns=[], num_records=0, project_id="local-preview-error"
            )

            error_results = PreviewResults(
                aidd_metadata=aidd_metadata,
                output=None,
                success=False,
                evaluation_results={"error": f"Local execution failed: {str(e)}"},
            )
            
            # Finalize debug file if streaming (even on error)
            if debug_file and executor.debug_file:
                executor._finalize_debug_file()
            
            # Add debug info to error results if enabled
            if debug_storage and executor.debug_storage:
                error_results.debug_info = executor.get_debug_summary()
                logger.info(f"🔍 Debug info: {error_results.debug_info['total_calls']} LLM calls captured (with errors)")
            
            return error_results

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

    def save_debug_info(self, filename: str = "llm_debug_calls.json"):
        """Save debug information from the last preview call."""
        if hasattr(self, '_last_executor') and self._last_executor and self._last_executor.debug_storage:
            self._last_executor.save_debug_calls(filename)
        else:
            logger.warning("No debug information available (debug_storage was not enabled or no preview was run)")

    def get_debug_summary(self) -> Dict[str, Any]:
        """Get debug summary from the last preview call."""
        if hasattr(self, '_last_executor') and self._last_executor and self._last_executor.debug_storage:
            return self._last_executor.get_debug_summary()
        else:
            return {"total_calls": 0, "successful_calls": 0, "failed_calls": 0}

    def patched_preview(
        self, verbose_logging: bool = False, validate: bool = True, debug_storage: bool = False, debug_file: Optional[str] = None
    ) -> PreviewResults:
        """Patched preview method that bypasses Gretel servers for custom connections."""
        if validate:
            self._run_semantic_validation(raise_exceptions=True)

        # Check if this configuration uses custom model connections
        if self._has_custom_connections():
            logger.info("🔄 Detected custom model connections - executing locally")
            return self._execute_local_preview(verbose_logging=verbose_logging, debug_storage=debug_storage, debug_file=debug_file)
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
    DataDesigner.save_debug_info = save_debug_info
    DataDesigner.get_debug_summary = get_debug_summary
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
