import React, { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';
import Button from './Button';

const ImageTransferLearning = ({ classes, setModel, setTrainingStatus, trainSettings }) => {
  const [baseModel, setBaseModel] = useState(null);
  const [isTraining, setIsTraining] = useState(false);

  useEffect(() => {
    const loadModel = async () => {
      try {
        const model = await mobilenet.load({
          version: 2,
          alpha: 1.0
        });
        setBaseModel(model);
        console.log('MobileNet V2 모델이 로드되었습니다.');
      } catch (error) {
        console.error('모델 로드 오류:', error);
      }
    };

    loadModel();
  }, []);

  const imageToTensor = (imageData) => {
    return tf.tidy(() => {
      const tensor = tf.browser.fromPixels(imageData)
        .resizeNearestNeighbor([224, 224])
        .toFloat();
      
      const offset = tf.scalar(127.5);
      const normalized = tensor.sub(offset).div(offset);
      return normalized.expandDims();
    });
  };

  const prepareData = async () => {
    const datasets = {
      xs: [],
      ys: []
    };

    try {
      for (let i = 0; i < classes.length; i++) {
        for (const sample of classes[i].samples) {
          const img = new Image();
          img.src = sample;
          await new Promise(resolve => {
            img.onload = resolve;
          });

          const features = await tf.tidy(() => {
            const tensor = imageToTensor(img);
            const activation = baseModel.infer(tensor, { embedding: true });
            return activation;
          });

          if (features) {
            datasets.xs.push(features);
            datasets.ys.push(i);
          }
        }
      }

      if (datasets.xs.length === 0) {
        throw new Error('유효한 샘플이 없습니다.');
      }

      const xs = tf.concat(datasets.xs);
      const ys = tf.oneHot(datasets.ys, classes.length);

      return { xs, ys };
    } catch (error) {
      console.error('데이터 준비 오류:', error);
      throw error;
    }
  };

  const trainModel = async () => {
    if (!baseModel || classes.every(c => c.samples.length === 0)) return;

    setIsTraining(true);
    setTrainingStatus({ epoch: 0, loss: 0, accuracy: 0 });

    try {
      const { xs, ys } = await prepareData();
      
      const newModel = tf.sequential();
      newModel.add(tf.layers.dense({
        units: 128,
        activation: 'relu',
        inputShape: [1280]
      }));
      newModel.add(tf.layers.dropout(0.5));
      newModel.add(tf.layers.dense({
        units: classes.length,
        activation: 'softmax'
      }));

      newModel.compile({
        optimizer: tf.train.adam(trainSettings.learningRate),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
      });

      await newModel.fit(xs, ys, {
        epochs: trainSettings.epochs,
        batchSize: trainSettings.batchSize,
        validationSplit: 0.2,
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            setTrainingStatus({
              epoch: epoch + 1,
              loss: logs.loss,
              accuracy: logs.acc
            });
            await tf.nextFrame();
          }
        }
      });

      setModel(newModel);
    } catch (error) {
      console.error('훈련 오류:', error);
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <div>
      <Button
        onClick={trainModel}
        disabled={isTraining || !baseModel}
        className="w-full"
      >
        {isTraining ? '훈련 중...' : '모델 학습시키기'}
      </Button>
    </div>
  );
};

export default ImageTransferLearning; 