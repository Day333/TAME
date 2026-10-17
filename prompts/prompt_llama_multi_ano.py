BASE_SYSTEM_PROMPT = "You are a helpful assistant."

# "Local Structural Jump Anomaly"删掉
ANOMALY_DICT = {
  "Normal Sequence": (0, "There are no abnormal situations in this time series. All variables exhibit expected behavior and relationships."),
  "Covariance Structure Anomaly": (1, "The stable covariance or correlation structure between variables unexpectedly changes, such as a flip in correlation direction."),
  "Temporal Dependency Anomaly": (2, "The usual temporal dependencies (e.g., fixed lags between variables) are violated, indicating timing disruptions."),
  "Trend Divergence Anomaly": (3, "Variables that usually follow similar trends begin to diverge unexpectedly, suggesting localized or partial faults."),
  "Joint Space Anomaly": (4, "The combination of values across multiple variables is anomalous, even though each variable appears normal in isolation."),
  "Principal Component Space Anomaly": (5, "An anomaly is revealed only in the reduced-dimensional space (e.g., PCA), indicating latent multivariate structural deviation."),
  "Collinearity Shift Anomaly": (6, "Strong linear relationships or redundancy between variables break down, suggesting sensor malfunction or desynchronization."),
}


USER_DETECTION_PROMPT = """You are an expert in multivariate time series anomaly detection. We provide a multivariate time series (called Observation), where each time point contains multiple variables. Your task is to identify the anomaly type (called Action) and provide detailed reasoning (called Thought).  

The reasoning (Thought) should analyze the relationships, dynamics, and structures across all variables and time points to infer any abnormal behavior.  

The Action must be one of the following seven types, where type 0 means no anomaly. The definitions are:

0. Normal Sequence: All variables follow expected behavior over time. Relationships among variables and their dynamics remain stable without any abnormality.
1. Covariance Structure Anomaly: The usual covariance or correlation structure among variables changes suddenly, such as reversal or unexpected decorrelation.
2. Temporal Dependency Anomaly: Expected temporal dependencies (e.g., fixed lags, response delays between variables) are violated, indicating possible desynchronization or timing failures.
3. Trend Divergence Anomaly: A subset of variables unexpectedly deviates from a shared trend, suggesting localized failures or partial system faults.
4. Joint Space Anomaly: Although individual variable values may appear normal, their joint configuration is anomalous—suggesting system-level inconsistency in the multivariate space.
5. Principal Component Space Anomaly: An anomaly becomes evident only in a lower-dimensional latent space (e.g., PCA), revealing subtle structural deviation across many variables.
6. Collinearity Shift Anomaly: Strong linear dependencies or redundancies between variables suddenly break down, often due to malfunctioning or desynchronized components.

The anomaly detection of each time series is divided into three steps: Observation, Thought, Action. After analyzing each observation, please provide the next Thought and next Action. Here are some examples:

Observation: Channel A: [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4];	Channel B: [1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5]; Channel C: [0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3]
Thought: All three channels exhibit a consistent and smooth upward trend with strong mutual correlation and regular spacing. There are no abrupt changes, outliers, timing issues, or structural shifts. The system behaves as expected, with clean temporal progression and stable multivariate relationships.
Action: Normal Sequence

Observation: Channel A: [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]; Channel B: [1.0, 1.6, 2.1, 2.7, 3.2, 2.5, 2.0, 1.5, 1.0, 0.5]
Thought: For the first five time points, A and B exhibit strong positive correlation. However, starting at time step 6, A continues increasing while B begins decreasing. This abrupt flip in correlation suggests a disruption in the system’s internal balance—possibly due to component failure, external interference, or an unexpected interaction between subsystems.
Action: Covariance Structure Anomaly

Observation: Channel A: [0, 0, 1, 0, 0, 1, 0, 0];	Channel B: [0, 0, 0, 0, 0, 0, 0, 1]
Thought: Normally, B responds two steps after A is activated (e.g., Channel B (expected delay = 2): [0, 0, 0, 1, 0, 0, 0, 1]). At the 6th time point, A activates again, but B now takes four steps to respond. Such an unexpected change in response timing may indicate communication delays, sensor lag, or disturbances in feedback control mechanisms.
Action: Temporal Dependency Anomaly

Observation: Channel A: [1.0, 1.2, 1.4, 1.6, 1.5, 1.3, 1.1, 0.9]; Channel B: [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
Thought: A and B move together initially with an upward trend. However, after time step 4, A reverses and begins decreasing while B continues rising. This trend divergence may point to a localized fault, an environmental disturbance affecting only one variable, or a control loop acting on just part of the system.
Action: Trend Divergence Anomaly

Observation: Temperature (°C): [70, 72, 73, 75, 80]; Pressure (kPa): [200, 210, 215, 220, 150]
Thought: Temperature and pressure values are both within acceptable individual ranges. However, historically, high temperatures have always been accompanied by high pressure. At time step 5, this joint relationship breaks down—indicating a hidden abnormality in the system’s internal coupling or thermodynamic behavior.
Action: Joint Space Anomaly

Observation: X: [10, 12, 11, 13, 15]; Y: [8, 9, 10, 11, 7]; Z: [5, 5.2, 5.1, 5.3, 9.0]
Thought: In this sequence, the PCA projection shows the fifth point as an outlier in the PC1-PC2 plane. Although all variables individually seem within range, the fifth point is far from the main cluster in the PCA-reduced space. This suggests a deviation in the latent multivariate structure of the system, which could be caused by complex, non-obvious interactions among features.
Action: Principal Component Space Anomaly

Observation: Sensor A: [100, 100, 100, 80, 100]; Sensor B: [100, 100, 100, 100, 100]
Thought: Sensor A and B are redundant and typically report identical values. At time step 4, A drops to 80 while B remains at 100. This sudden deviation violates expected collinearity, possibly indicating sensor malfunction, desynchronization, or unexpected behavior in one part of the system.
Action: Collinearity Shift Anomaly


Here is a multivariate time series observation that we need to check for anomaly categories. The oberservation is from the domain of {our_source}.
Please make a Thought judgment and put your final Action within \\boxed1{{}} and \\boxed2{{}} respectively, where action must just be a category name not id.
Observation: {our_observation}
Thought: \\boxed1{{}}
Action: \\boxed2{{}}
"""

# 背景:两个标注脚本都调用了 USER_DETECTION_LABELED_PROMPT,但开源版把这个变量连同它所在的文件一起剔掉了,运行时直接 NameError。

# 说明:评测脚本 eval_*.py 用的是不带标签的 USER_DETECTION_PROMPT(评测时不能把答案喂给模型)

USER_DETECTION_LABELED_PROMPT = USER_DETECTION_PROMPT.replace(
    "Here is a multivariate time series observation that we need to check for "
    "anomaly categories. The oberservation is from the domain of {our_source}.",
    "Here is a multivariate time series observation that we need to check for "
    "anomaly categories. We already know that it is a {our_label} sequence and "
    "from the domain of {our_source}.",
)

# 防呆:若将来原文措辞被改动,replace 会静默失效(退化成和 USER_DETECTION_PROMPT一模一样,模型就收不到已知标签)。这里主动报错提醒。
assert USER_DETECTION_LABELED_PROMPT != USER_DETECTION_PROMPT, (
    "USER_DETECTION_LABELED_PROMPT substitution failed: the wording of "
    "USER_DETECTION_PROMPT has changed. Please update the search string above."
)
