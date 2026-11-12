//! Builds the RunReady configuration that tells Cursor's client-side agent how to prompt a BYOK model.
use serde_json::Value;

use crate::{cursor::protocol::proto::inference as pb, Result};

/// Cursor's own prompt configuration for models it does not classify (`unclassified` in the
/// agent host's bundled prompt-model table), used verbatim for every BYOK model.
const UNCLASSIFIED: &str = include_str!("prompt_metadata.json");

/// Flags the managed local loop needs that the `unclassified` table leaves off, copied from the
/// agent host's own pinned local-loop defaults. The local loop cannot build in-process child
/// agents, so the Task tool fails ("managed local loop does not build in-process child
/// AgentConfig") unless subagents run through the client-side executor, which starts each child
/// as its own RunInference stream.
const FLAG_OVERRIDES: &[(&str, bool)] = &[
    ("useClientSideSubagent", true),
    ("enableExploreSubagent", true),
    ("enableDebugSubagent", true),
    ("enableAwaitForSubagents", true),
    ("enableBoundedSubagentAwait", true),
    ("enableBackgroundTaskProgress", true),
    ("subagentSupportInterrupt", true),
    ("longRunningJobs", true),
];

/// `context_window_tokens` becomes the client's `agent_token_limit`: the budget Cursor's agent
/// compacts against (it summarizes with the same model through this stream). Without it Cursor
/// falls back to its own default window.
///
/// `overrides` fills in `subagent_model_config.overrides` so the client honors the per-subagent
/// model choices from the RunRequest (via Cursor's Task tool UI); without them every subagent is
/// pinned to the parent model.
pub fn run_ready(
    requested: &pb::InferenceRequestedModel,
    display_name: &str,
    context_window_tokens: Option<u64>,
    overrides: Vec<pb::RunInferenceResolvedSubagentModelOverride>,
) -> Result<pb::RunInferenceRunReady> {
    let mut metadata = prompt_model_metadata(display_name, &requested.model_id, overrides)?;
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

fn prompt_model_metadata(
    display_name: &str,
    parent_model_id: &str,
    overrides: Vec<pb::RunInferenceResolvedSubagentModelOverride>,
) -> Result<pb::RunInferencePromptModelMetadata> {
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
        subagent_model_config: subagent_config(parent_model_id, overrides),
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
            .chain(FLAG_OVERRIDES.iter().map(|(name, value)| {
                (
                    (*name).to_owned(),
                    pb::RunInferenceFlagValue {
                        value: Some(pb::run_inference_flag_value::Value::BoolValue(*value)),
                    },
                )
            }))
            .collect(),
        ..Default::default()
    })
}

fn subagent_config(
    parent_model_id: &str,
    overrides: Vec<pb::RunInferenceResolvedSubagentModelOverride>,
) -> Option<pb::RunInferenceSubagentModelConfig> {
    if overrides.is_empty() {
        return None;
    }
    // The client treats any slug missing from `models` as invalid and falls back to the parent,
    // so every model the user picked (and the parent itself) is advertised as valid here.
    let mut slugs = vec![parent_model_id.to_owned()];
    for model_id in overrides.iter().filter_map(|value| value.model_id.as_ref()) {
        if !slugs.contains(model_id) {
            slugs.push(model_id.clone());
        }
    }
    Some(pb::RunInferenceSubagentModelConfig {
        force_policy: "none".into(),
        parent_requested_model_name: parent_model_id.into(),
        models: slugs
            .into_iter()
            .map(|slug| pb::RunInferenceSubagentModelPolicy {
                slug,
                advertised: true,
                valid: true,
                ..Default::default()
            })
            .collect(),
        overrides,
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
            Vec::new(),
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
        assert_eq!(
            metadata.feature_flags["useClientSideSubagent"].value,
            Some(pb::run_inference_flag_value::Value::BoolValue(true))
        );
        assert_eq!(ready.resolved_model.unwrap().model_id, "byok-model");
    }

    #[test]
    fn subagent_config_empty_when_no_overrides() {
        let ready = run_ready(
            &pb::InferenceRequestedModel {
                model_id: "parent".into(),
                ..Default::default()
            },
            "Parent",
            None,
            Vec::new(),
        )
        .unwrap();
        let metadata = ready.prompt_model_metadata.unwrap();
        assert!(metadata.subagent_model_config.is_none());
    }

    #[test]
    fn subagent_config_carries_overrides() {
        let ready = run_ready(
            &pb::InferenceRequestedModel {
                model_id: "parent".into(),
                ..Default::default()
            },
            "Parent",
            None,
            vec![pb::RunInferenceResolvedSubagentModelOverride {
                subagent_type: "explore".into(),
                kind: "model".into(),
                model_id: Some("fast-model".into()),
            }],
        )
        .unwrap();
        let metadata = ready.prompt_model_metadata.unwrap();
        let config = metadata.subagent_model_config.unwrap();
        assert_eq!(config.force_policy, "none");
        assert_eq!(config.parent_requested_model_name, "parent");
        assert_eq!(config.overrides.len(), 1);
        assert_eq!(config.overrides[0].subagent_type, "explore");
        assert_eq!(config.overrides[0].model_id.as_deref(), Some("fast-model"));
        let slugs: Vec<_> = config.models.iter().map(|model| model.slug.as_str()).collect();
        assert_eq!(slugs, ["parent", "fast-model"]);
        assert!(config.models.iter().all(|model| model.valid && model.advertised));
    }
}
