//! Builds the RunReady configuration that tells Cursor's client-side agent how to prompt a BYOK model.
use serde_json::Value;

use crate::{cursor::protocol::proto::inference as pb, Result};

/// Cursor's own prompt configuration for models it does not classify (`unclassified` in the
/// agent host's bundled prompt-model table), used verbatim for every BYOK model.
const UNCLASSIFIED: &str = include_str!("prompt_metadata.json");

/// `context_window_tokens` becomes the client's `agent_token_limit`: the budget Cursor's agent
/// compacts against (it summarizes with the same model through this stream). Without it Cursor
/// falls back to its own default window.
pub fn run_ready(
    requested: &pb::InferenceRequestedModel,
    display_name: &str,
    context_window_tokens: Option<u64>,
) -> Result<pb::RunInferenceRunReady> {
    let mut metadata = prompt_model_metadata(display_name)?;
    metadata.agent_token_limit =
        context_window_tokens.map(|tokens| tokens.min(i32::MAX as u64) as i32);
    Ok(pb::RunInferenceRunReady {
        resolved_model: Some(requested.clone()),
        routed_model_display_name: Some(display_name.to_owned()),
        prompt_model_metadata: Some(metadata),
        accepts_aborted_finish: true,
        ..Default::default()
    })
}

fn prompt_model_metadata(display_name: &str) -> Result<pb::RunInferencePromptModelMetadata> {
    let config: Value = serde_json::from_str(UNCLASSIFIED)?;
    let text = |key: &str| config[key].as_str().unwrap_or_default().to_owned();
    let flag = |key: &str| config[key].as_bool().unwrap_or_default();
    Ok(pb::RunInferencePromptModelMetadata {
        vendor: text("vendor"),
        prompt_version: text("promptVersion"),
        persona: text("persona"),
        self_identity_name: display_name.to_owned(),
        supports_assistant_message_prefill: flag("supportsAssistantMessagePrefill"),
        enable_line_numbers: flag("enableLineNumbers"),
        use_sparse_read_line_numbers: flag("useSparseReadLineNumbers"),
        dsv3_is_thinking: Some(flag("dsv3IsThinking")),
        enable_semantic_search: Some(flag("enableSemanticSearch")),
        use_new_plan_mode_prompts: Some(flag("useNewPlanModePrompts")),
        estimated_cache_ttl_ms: config["estimatedCacheTtlMs"].as_i64().map(|ms| ms as i32),
        // The classifier routes through Cursor-hosted models, which BYOK turns cannot reach.
        smart_mode_classifier_mode: Some(false),
        smart_mode_classifier_shadow_mode: Some(false),
        agent_mode_configs: config["agentModeConfigs"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|mode| mode["modeId"].as_str())
            .map(|mode_id| pb::RunInferenceAgentModeConfig {
                mode_id: mode_id.to_owned(),
                ..Default::default()
            })
            .collect(),
        ask_question_config: config["askQuestionConfig"]["toolName"]
            .as_str()
            .map(|tool_name| pb::RunInferenceAskQuestionConfig {
                tool_name: tool_name.to_owned(),
                auto_reject_first: false,
            }),
        feature_flags: config["featureFlags"]
            .as_object()
            .into_iter()
            .flatten()
            .filter_map(|(name, value)| Some((name.clone(), flag_value(value)?)))
            .collect(),
        ..Default::default()
    })
}

fn flag_value(value: &Value) -> Option<pb::RunInferenceFlagValue> {
    use pb::run_inference_flag_value::Value as Flag;
    let flag = if let Some(flag) = value["boolValue"].as_bool() {
        Flag::BoolValue(flag)
    } else if let Some(number) = value["numberValue"].as_f64() {
        Flag::NumberValue(number)
    } else {
        Flag::StringValue(value["stringValue"].as_str()?.to_owned())
    };
    Some(pb::RunInferenceFlagValue { value: Some(flag) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_unclassified_metadata_for_byok_models() {
        let ready = run_ready(
            &pb::InferenceRequestedModel {
                model_id: "byok-model".into(),
                ..Default::default()
            },
            "My Model",
            Some(272_000),
        )
        .unwrap();
        let metadata = ready.prompt_model_metadata.unwrap();
        assert_eq!(metadata.vendor, "unknown");
        assert_eq!(metadata.agent_token_limit, Some(272_000));
        assert_eq!(metadata.self_identity_name, "My Model");
        assert!(metadata
            .agent_mode_configs
            .iter()
            .any(|mode| mode.mode_id == "agent"));
        assert!(metadata.feature_flags.len() > 100);
        assert_eq!(ready.resolved_model.unwrap().model_id, "byok-model");
    }
}
