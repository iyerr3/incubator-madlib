DROP TABLE IF EXISTS mnist_result_summary;
DROP TABLE IF EXISTS mnist_result;
SELECT madlib.mlp_classification(
    'mnist_train',
    'mnist_result',
    'x',
    'y',
    ARRAY[5],
    'step_size=0.0005,
    n_iterations=100,
    tolerance=0','sigmoid');

DROP TABLE IF EXISTS mnist_test_prediction;
SELECT madlib.mlp_predict(
    'mnist_result',
    'mnist_test',
    'id',
    'mnist_test_prediction',
    'response');
DROP TABLE IF EXISTS mnist_train_prediction;
SELECT madlib.mlp_predict(
    'mnist_result',
    'mnist_train',
    'id',
    'mnist_train_prediction',
    'response');

SELECT CONCAT(round(count(*)*100/60000.0,2),'%') as train_accuracy from
    (select mnist_train.y as a, mnist_train_prediction.estimated_y as b from mnist_train_prediction inner join mnist_train on mnist_train.id=mnist_train_prediction.id) q
WHERE q.a=q.b;

SELECT CONCAT(round(count(*)*100/10000.0,2),'%') as test_accuracy from
    (select mnist_test.y as a, mnist_test_prediction.estimated_y as b from mnist_test_prediction inner join mnist_test on mnist_test.id=mnist_test_prediction.id) q
WHERE q.a=q.b;
