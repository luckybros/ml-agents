using System;
using UnityEngine;

namespace Unity.MLAgents.SideChannels
{
    /// <summary>
    /// Side channel that supports modifying attributes specific to the Unity Engine.
    /// </summary>
    internal class EngineConfigurationChannel : SideChannel
    {
        internal enum ConfigurationType : int
        {
            ScreenResolution = 0,
            QualityLevel = 1,
            TimeScale = 2,
            TargetFrameRate = 3,
            CaptureFrameRate = 4
        }

        const string k_EngineConfigId = "e951342c-4f7e-11ea-b238-784f4387d1f7";

        /// <summary>
        /// Initializes the side channel. The constructor is internal because only one instance is
        /// supported at a time, and is created by the Academy.
        /// </summary>
        internal EngineConfigurationChannel()
        {
            ChannelId = new Guid(k_EngineConfigId);
        }

        /// <inheritdoc/>
        protected override void OnMessageReceived(IncomingMessage msg)
        {
            var messageType = (ConfigurationType)msg.ReadInt32();
            switch (messageType)
            {
                case ConfigurationType.ScreenResolution:
                    var width = msg.ReadInt32();
                    var height = msg.ReadInt32();
                    Debug.Log($"[EngineConfig] ScreenResolution impostata a: {width}x{height}");
                    Screen.SetResolution(width, height, false);
                    break;
                case ConfigurationType.QualityLevel:
                    var qualityLevel = msg.ReadInt32();
                    Debug.Log($"[EngineConfig] QualityLevel impostato a: {qualityLevel}");
                    QualitySettings.SetQualityLevel(qualityLevel, true);
                    break;
                case ConfigurationType.TimeScale:
                    var timeScale = msg.ReadFloat32();

                    // There's an upper limit for the timeScale in the editor (but not in the player)
                    // Always ensure that timeScale >= 1 also,
#if UNITY_EDITOR
                    const float maxTimeScale = 100f;
#else
                    const float maxTimeScale = float.PositiveInfinity;
#endif
                    timeScale = Mathf.Clamp(timeScale, 1, maxTimeScale);
                    Debug.Log($"[TimeScale] Time scale impostato a {timeScale}");
                    Time.timeScale = timeScale;
                    break;
                case ConfigurationType.TargetFrameRate:
                    var targetFrameRate = msg.ReadInt32();
                    Debug.Log($"[EngineConfig] TargetFrameRate impostato a: {targetFrameRate}");
                    Application.targetFrameRate = targetFrameRate;
                    break;
                case ConfigurationType.CaptureFrameRate:
                    var captureFrameRate = msg.ReadInt32();
                    Debug.Log($"[EngineConfig] CaptureFrameRate impostato a: {captureFrameRate}");
                    Time.captureFramerate = captureFrameRate;
                    break;
                default:
                    Debug.LogWarning(
                        "Unknown engine configuration received from Python. Make sure" +
                        " your Unity and Python versions are compatible.");
                    break;
            }
        }
    }
}
