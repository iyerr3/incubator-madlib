DROP TABLE IF EXISTS mlp_class;
DROP TABLE IF EXISTS mlp_class_summary;
SELECT madlib.mlp_classification(
    'iris_data',      -- Source table
    'mlp_class',    -- Desination table
    'attributes',     -- Input features
    'class_text',   -- Label
    ARRAY[5],     -- Number of units per layer
    'step_size=0.001,
    n_iterations=1000,
    tolerance=0',
    'tanh');

SELECT * FROM mlp_class;
SELECT * FROM mlp_class_summary;
DROP TABLE IF EXISTS mlp_prediction;
SELECT setseed(0);
SELECT madlib.mlp_predict(
    'mlp_class',
    'iris_data',
    'id',
    'mlp_prediction',
    'response');
-- SELECT * FROM mlp_prediction;
SELECT (
    COUNT(*)/150.0
) as accuracy FROM
    (SELECT iris_data.class_text AS actual, mlp_prediction.estimated_class_text as estimated
    FROM mlp_prediction INNER JOIN iris_data ON
    iris_data.id=mlp_prediction.id) q
WHERE q.actual=q.estimated;

