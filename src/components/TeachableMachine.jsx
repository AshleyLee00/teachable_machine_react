import React, { useState, useRef, useEffect, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as tfvis from '@tensorflow/tfjs-vis';

const TeachableMachine = () => {
  const [classes, setClasses] = useState([
    { name: 'Class 1', samples: [] },
    { name: 'Class 2', samples: [] }
  ]);
  const [activeClassIndex, setActiveClassIndex] = useState(null);
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [model, setModel] = useState(null);
  const [baseModel, setBaseModel] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainSettings, setTrainSettings] = useState({
    epochs: 50,
    batchSize: 32,
    learningRate: 0.0001,
  });
  const [trainStatus, setTrainStatus] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isPreviewActive, setIsPreviewActive] = useState(false);
  const [previewPrediction, setPreviewPrediction] = useState(null);
  const previewVideoRef = useRef(null);
  const previewStreamRef = useRef(null);
  
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const fileInputRefs = useRef([]);

  // 훈련 상태를 위한 상태 추가
  const [trainingStatus, setTrainingStatus] = useState({
    epoch: 0,
    loss: 0,
    accuracy: 0
  });

  useEffect(() => {
    fileInputRefs.current = classes.map(() => React.createRef());
  }, [classes]);

  // MobileNet 모델 로드
  useEffect(() => {
    const loadMobileNet = async () => {
      const mobilenetModel = await mobilenet.load();
      setBaseModel(mobilenetModel);
      console.log('MobileNet 모델이 로드되었습니다.');
    };
    loadMobileNet();
  }, []);

  // 이미지를 텐서로 변환하는 함수
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

  // 데이터셋 준비
  const prepareData = async () => {
    const datasets = {
      xs: [],
      ys: []
    };

    for (let i = 0; i < classes.length; i++) {
      for (const sample of classes[i].samples) {
        const img = new Image();
        img.src = sample;
        await new Promise(resolve => {
          img.onload = resolve;
        });

        const tensor = imageToTensor(img);
        const activation = baseModel.infer(tensor, true);
        datasets.xs.push(activation);
        datasets.ys.push(i);
      }
    }

    return {
      xs: tf.concat(datasets.xs),
      ys: tf.oneHot(datasets.ys, classes.length)
    };
  };

  // 모델 훈련
  const trainModel = async () => {
    if (!baseModel || classes.every(c => c.samples.length === 0)) return;

    setIsTraining(true);
    setTrainingStatus({ epoch: 0, loss: 0, accuracy: 0 });

    try {
      // 데이터 준비
      const { xs, ys } = await prepareData();
      
      // 모델 설정
      const layer = baseModel.getLayer('conv_pw_13_relu');
      const newModel = tf.sequential({
        layers: [
          tf.layers.globalAveragePooling2d({ inputShape: layer.outputShape.slice(1) }),
          tf.layers.dense({ units: classes.length, activation: 'softmax' })
        ]
      });

      // 컴파일
      newModel.compile({
        optimizer: tf.train.adam(trainSettings.learningRate),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
      });

      // 훈련 시작
      await newModel.fit(xs, ys, {
        epochs: trainSettings.epochs,
        batchSize: trainSettings.batchSize,
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            // 실시간으로 훈련 상태 업데이트
            setTrainingStatus({
              epoch: epoch + 1,
              loss: logs.loss,
              accuracy: logs.acc
            });
            // UI 업데이트를 위한 지연
            await tf.nextFrame();
          }
        }
      });

      setModel(newModel);
    } catch (error) {
      console.error('훈련 오류:', error);
      alert('모델 훈련 중 오류가 발생했습니다.');
    } finally {
      setIsTraining(false);
    }
  };

  // 모델 내보내기
  const exportModel = async () => {
    if (!model) return;

    try {
      await model.save('downloads://model');
    } catch (error) {
      console.error('모델 내보내기 오류:', error);
      alert('모델 내보내기 중 오류가 발생했습니다.');
    }
  };

  // 성능 최적화를 위한 상수 정의
  const PREDICTION_INTERVAL = 100; // 예측 간격 (ms)
  const IMAGE_SIZE = 224; // 모델 입력 크기

  // predict 함수 최적화
  const predict = async (image) => {
    if (!model) return null;

    try {
      // 메모리 최적화를 위해 tidy 사용
      return await tf.tidy(() => {
        const tensor = tf.browser.fromPixels(image)
          .resizeNearestNeighbor([IMAGE_SIZE, IMAGE_SIZE])
          .toFloat()
          .div(255.0)  // 정규화
          .expandDims();
        
        const predictions = model.predict(tensor);
        const probabilities = predictions.dataSync();
        const maxProbability = Math.max(...probabilities);
        const classIndex = probabilities.indexOf(maxProbability);

        return {
          className: classes[classIndex].name,
          probability: maxProbability
        };
      });
    } catch (error) {
      console.error('예측 오류:', error);
      return null;
    }
  };

  // 실시간 예측 효과 추가
  useEffect(() => {
    let animationFrameId;

    const runPrediction = async () => {
      if (isPreviewActive && previewVideoRef.current && model && baseModel) {
        const prediction = await predict(previewVideoRef.current);
        if (prediction) {
          setPreviewPrediction(prediction);
        }
        animationFrameId = requestAnimationFrame(runPrediction);
      }
    };

    if (isPreviewActive) {
      runPrediction();
    }

    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, [isPreviewActive, model, baseModel, predict]);

  // JSX에 추가할 훈련 설정 UI
  const renderTrainingControls = () => (
    <div className="bg-white rounded-lg shadow p-4 mb-4">
      <h3 className="font-bold mb-4">훈련 설정</h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <label className="block text-sm font-medium mb-1">에포크 (Epochs)</label>
          <input
            type="number"
            value={trainSettings.epochs}
            onChange={(e) => setTrainSettings(prev => ({
              ...prev,
              epochs: parseInt(e.target.value)
            }))}
            className="w-full p-2 border rounded"
            min="1"
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">배치 크기</label>
          <input
            type="number"
            value={trainSettings.batchSize}
            onChange={(e) => setTrainSettings(prev => ({
              ...prev,
              batchSize: parseInt(e.target.value)
            }))}
            className="w-full p-2 border rounded"
            min="1"
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">학습률</label>
          <input
            type="number"
            value={trainSettings.learningRate}
            onChange={(e) => setTrainSettings(prev => ({
              ...prev,
              learningRate: parseFloat(e.target.value)
            }))}
            className="w-full p-2 border rounded"
            step="0.0001"
          />
        </div>
      </div>
      <div className="mt-4 flex justify-between items-center">
        <button
          onClick={trainModel}
          disabled={isTraining || !baseModel}
          className={`bg-blue-500 text-white px-4 py-2 rounded ${
            (isTraining || !baseModel) ? 'opacity-50 cursor-not-allowed' : ''
          }`}
        >
          {isTraining ? '훈련 중...' : '모델 훈련 시작'}
        </button>
        <button
          onClick={exportModel}
          disabled={!model}
          className={`bg-green-500 text-white px-4 py-2 rounded ${
            !model ? 'opacity-50 cursor-not-allowed' : ''
          }`}
        >
          모델 내보내기
        </button>
      </div>
      {trainStatus && (
        <div className="mt-2 text-sm text-gray-600">
          {trainStatus}
        </div>
      )}
    </div>
  );

  const startWebcam = async (classIndex) => {
    try {
      setIsWebcamActive(true);
      setActiveClassIndex(classIndex);

      await new Promise(resolve => setTimeout(resolve, 100));

      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 1280 },
          height: { ideal: 720 }
        } 
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play().catch(e => {
            console.error('비디오 재생 오류:', e);
          });
        };
        streamRef.current = stream;
      } else {
        stream.getTracks().forEach(track => track.stop());
        setIsWebcamActive(false);
        setActiveClassIndex(null);
        throw new Error('비디오 엘리먼트를 찾을 수 없습니다.');
      }
    } catch (err) {
      console.error('카메라 접근 오류:', err);
      setIsWebcamActive(false);
      setActiveClassIndex(null);
      alert('카메라를 시작할 수 없습니다.');
    }
  };

  const stopWebcam = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      setIsWebcamActive(false);
      setActiveClassIndex(null);
    }
  };

  const captureImage = () => {
    if (!videoRef.current) return;

    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoRef.current, 0, 0);
    
    const imageData = canvas.toDataURL('image/jpeg');
    addImageToClass(activeClassIndex, imageData);
  };

  const handleFileUpload = async (e, classIndex) => {
    const files = Array.from(e.target.files);
    
    for (const file of files) {
      if (!file.type.startsWith('image/')) {
        continue;
      }

      try {
        const imageUrl = URL.createObjectURL(file);
        
        setClasses(prevClasses => {
          const newClasses = [...prevClasses];
          newClasses[classIndex] = {
            ...newClasses[classIndex],
            samples: [...newClasses[classIndex].samples, imageUrl]
          };
          return newClasses;
        });
      } catch (error) {
        console.error('이미지 업로드 오류:', error);
      }
    }
  };

  const addImageToClass = (classIndex, imageData) => {
    setClasses(prev => prev.map((c, i) => {
      if (i === classIndex) {
        return {
          ...c,
          samples: [...c.samples, imageData]
        };
      }
      return c;
    }));
  };

  const removeImage = (classIndex, imageIndex) => {
    setClasses(prev => prev.map((c, i) => {
      if (i === classIndex) {
        return {
          ...c,
          samples: c.samples.filter((_, imgI) => imgI !== imageIndex)
        };
      }
      return c;
    }));
  };

  const addClass = () => {
    setClasses(prev => [...prev, {
      name: `Class ${prev.length + 1}`,
      samples: []
    }]);
  };

  const removeClass = (indexToRemove) => {
    setClasses(prev => prev.filter((_, index) => index !== indexToRemove));
  };

  const updateClassName = (index, newName) => {
    setClasses(prev => prev.map((c, i) => 
      i === index ? { ...c, name: newName } : c
    ));
  };

  // 미리보기 시작 함수
  const startPreview = async () => {
    try {
      // 이미 활성화된 스트림이 있다면 중지
      if (previewVideoRef.current?.srcObject) {
        stopPreview();
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user',
          frameRate: { ideal: 30 }  // 프레임레이트 설정
        }
      });

      if (previewVideoRef.current) {
        previewVideoRef.current.srcObject = stream;
        previewVideoRef.current.onloadeddata = () => {
          setIsPreviewActive(true);
          startPreviewPrediction();
        };
      }
    } catch (err) {
      console.error('웹캠 시작 오류:', err);
      alert('웹캠을 시작할 수 없습니다. 카메라 권한을 확인해주세요.');
    }
  };

  // 미리보기 중지 함수
  const stopPreview = useCallback(() => {
    if (previewVideoRef.current?.srcObject) {
      const tracks = previewVideoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      previewVideoRef.current.srcObject = null;
    }
    setIsPreviewActive(false);
    setPreviewPrediction(null);
    // 메모리 정리
    tf.disposeVariables();
  }, []);

  // 컴포넌트 언마운트 시 정리
  useEffect(() => {
    return () => {
      stopPreview();
    };
  }, [stopPreview]);

  // 예측 함수 최적화
  const startPreviewPrediction = () => {
    if (!model || !previewVideoRef.current) return;

    let lastPredictionTime = 0;
    let animationFrameId;
    let isProcessing = false;

    const predictFrame = async (timestamp) => {
      if (!isPreviewActive || !previewVideoRef.current) {
        cancelAnimationFrame(animationFrameId);
        return;
      }

      // 예측 간격 조절
      if (timestamp - lastPredictionTime > PREDICTION_INTERVAL && !isProcessing) {
        isProcessing = true;
        const videoElement = previewVideoRef.current;

        if (videoElement.readyState === videoElement.HAVE_ENOUGH_DATA) {
          try {
            const prediction = await predict(videoElement);
            if (prediction) {
              setPreviewPrediction(prediction);
            }
          } catch (error) {
            console.error('예측 오류:', error);
          }
          lastPredictionTime = timestamp;
        }
        isProcessing = false;
      }

      animationFrameId = requestAnimationFrame(predictFrame);
    };

    animationFrameId = requestAnimationFrame(predictFrame);

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  };

  // SVG 선 연결을 위한 스타일 추가
  const styles = {
    connector: {
      position: 'absolute',
      borderRight: '2px dashed #ccc',
      width: '50px',
      right: '-50px',
      top: '50%',
      transform: 'translateY(-50%)',
      height: '2px',
      backgroundColor: '#ccc'
    }
  };

  return (
    <div className="min-h-screen bg-gray-100">
      {/* 헤더 */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center">
          <h1 className="text-xl font-medium text-blue-600">Teachable Machine</h1>
        </div>
      </header>

      {/* 메인 컨텐츠 */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-12 gap-x-24 gap-y-6 relative">
          {/* 클래스 섹션 - 너비 축소 */}
          <div className="col-span-4 relative z-10">
            {classes.map((classData, classIndex) => (
              <div key={classIndex} className="mb-6 bg-white rounded-lg shadow relative">
                {/* 곡선 연결선 */}
                <svg 
                  className="absolute right-0 top-1/2 w-[100px] h-[40px]"
                  style={{
                    transform: 'translateY(-50%)',
                    right: '-100px',
                    zIndex: -1
                  }}
                >
                  <path
                    d="M0,20 C30,20 70,20 100,20"
                    stroke="#ccc"
                    strokeWidth="2"
                    strokeDasharray="4 4"
                    fill="none"
                  />
                </svg>
                
                {/* 기존 클래스 컨텐츠 */}
                <div className="p-4 border-b flex justify-between items-center">
                  <div className="flex items-center">
                    <input
                      type="text"
                      value={classData.name}
                      onChange={(e) => updateClassName(classIndex, e.target.value)}
                      className="text-lg font-medium bg-transparent border-none focus:outline-none"
                    />
                    <button className="ml-2 text-gray-400 hover:text-gray-600">
                      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M13.586 3.586a2 2 0 112.828 2.828l-.793.793-2.828-2.828.793-.793zM11.379 5.793L3 14.172V17h2.828l8.38-8.379-2.83-2.828z" />
                      </svg>
                    </button>
                  </div>
                  {classes.length > 2 && (
                    <button
                      onClick={() => removeClass(classIndex)}
                      className="text-gray-400 hover:text-gray-600"
                    >
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  )}
                </div>

                <div className="p-4">
                  <div className="text-sm text-gray-500 mb-4">이미지 샘플 추가:</div>
                  <div className="flex space-x-4">
                    <button
                      onClick={() => startWebcam(classIndex)}
                      className="flex-1 flex items-center justify-center px-4 py-2 border border-gray-300 rounded-md hover:bg-gray-50"
                    >
                      <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                      </svg>
                      웹캠
                    </button>
                    <input
                      type="file"
                      accept="image/*"
                      multiple
                      className="hidden"
                      ref={el => fileInputRefs.current[classIndex] = el}
                      onChange={(e) => handleFileUpload(e, classIndex)}
                      onClick={(e) => e.target.value = null}
                    />
                    <button
                      onClick={() => fileInputRefs.current[classIndex]?.click()}
                      className="flex-1 flex items-center justify-center px-4 py-2 border border-gray-300 rounded-md hover:bg-gray-50"
                    >
                      <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                      </svg>
                      업로드
                    </button>
                  </div>

                  {/* 샘플 이미지 그리드 */}
                  <div className="mt-4 grid grid-cols-6 gap-2">
                    {classData.samples.map((sample, imageIndex) => (
                      <div key={imageIndex} className="relative group aspect-w-1 aspect-h-1">
                        <img
                          src={sample}
                          alt={`Sample ${imageIndex + 1}`}
                          className="w-full h-full object-cover rounded"
                        />
                        <button
                          onClick={() => removeImage(classIndex, imageIndex)}
                          className="absolute top-1 right-1 bg-black bg-opacity-50 text-white rounded-full w-6 h-6 flex items-center justify-center opacity-0 group-hover:opacity-100"
                        >
                          ×
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ))}
            
            <button 
              onClick={addClass}
              className="w-full py-4 border-2 border-dashed border-gray-300 rounded-lg text-gray-500 hover:border-gray-400 hover:text-gray-600"
            >
              + Add New Class
            </button>
          </div>

          {/* 중앙 연결선 */}
          <div className="absolute left-1/2 top-0 bottom-0 w-0 -translate-x-1/2" style={{ zIndex: -1 }}>
            <svg 
              className="h-full w-[40px]"
              style={{ transform: 'translateX(-50%)' }}
            >
              <path
                d="M20,0 C20,200 20,800 20,1000"
                stroke="#ccc"
                strokeWidth="2"
                strokeDasharray="4 4"
                fill="none"
              />
            </svg>
          </div>

          {/* 훈련 설정 섹션 - 너비 확대 */}
          <div className="col-span-4 relative z-10">
            <div className="bg-white rounded-lg shadow">
              <div className="p-4 border-b">
                <h2 className="text-lg font-medium">학습</h2>
              </div>
              <div className="p-4">
                <button
                  onClick={trainModel}
                  disabled={isTraining || !baseModel}
                  className={`w-full py-2 px-4 rounded-md text-white ${
                    isTraining || !baseModel ? 'bg-gray-400' : 'bg-blue-600 hover:bg-blue-700'
                  }`}
                >
                  {isTraining ? '훈련 중...' : '모델 학습시키기'}
                </button>

                {/* 훈련 진행 상태 표시 */}
                {isTraining && (
                  <div className="mt-4 p-4 bg-blue-50 rounded-md">
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-blue-700">진행 상태:</span>
                        <span className="font-medium text-blue-800">
                          {trainingStatus.epoch} / {trainSettings.epochs} 에포크
                        </span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-blue-700">손실값:</span>
                        <span className="font-medium text-blue-800">
                          {trainingStatus.loss.toFixed(4)}
                        </span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-blue-700">정확도:</span>
                        <span className="font-medium text-blue-800">
                          {(trainingStatus.accuracy * 100).toFixed(1)}%
                        </span>
                      </div>
                      {/* 진행 바 */}
                      <div className="w-full bg-blue-200 rounded-full h-2.5">
                        <div 
                          className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
                          style={{ 
                            width: `${(trainingStatus.epoch / trainSettings.epochs) * 100}%` 
                          }}
                        />
                      </div>
                    </div>
                  </div>
                )}

                <div className="mt-6">
                  <h3 className="text-sm font-medium text-gray-900 mb-4">고급</h3>
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm text-gray-700">에포크:</label>
                      <div className="mt-1 relative">
                        <input
                          type="number"
                          value={trainSettings.epochs}
                          onChange={(e) => setTrainSettings(prev => ({
                            ...prev,
                            epochs: parseInt(e.target.value)
                          }))}
                          className="w-full px-3 py-2 border rounded-md"
                          min="1"
                        />
                      </div>
                    </div>
                    <div>
                      <label className="block text-sm text-gray-700">배치 크기:</label>
                      <div className="mt-1 relative">
                        <input
                          type="number"
                          value={trainSettings.batchSize}
                          onChange={(e) => setTrainSettings(prev => ({
                            ...prev,
                            batchSize: parseInt(e.target.value)
                          }))}
                          className="w-full px-3 py-2 border rounded-md"
                          min="1"
                        />
                      </div>
                    </div>
                    <div>
                      <label className="block text-sm text-gray-700">학습률:</label>
                      <div className="mt-1 relative">
                        <input
                          type="number"
                          value={trainSettings.learningRate}
                          onChange={(e) => setTrainSettings(prev => ({
                            ...prev,
                            learningRate: parseFloat(e.target.value)
                          }))}
                          className="w-full px-3 py-2 border rounded-md"
                          step="0.0001"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* 실시간 예측 및 내보내기 섹션 */}
          <div className="col-span-4 relative z-10 space-y-6">
            {/* 모델 테스트 */}
            <div className="bg-white rounded-lg shadow">
              <div className="p-4 border-b">
                <h2 className="text-lg font-medium">모델 테스트</h2>
              </div>
              <div className="p-4">
                {!model ? (
                  <div className="text-center text-gray-500 py-4">
                    먼저 모델을 학습시켜주세요.
                  </div>
                ) : !isPreviewActive ? (
                  <button
                    onClick={startPreview}
                    className="w-full py-2 px-4 bg-green-600 text-white rounded-md hover:bg-green-700"
                  >
                    웹캠으로 테스트하기
                  </button>
                ) : (
                  <div className="relative w-full aspect-video">
                    <video
                      ref={previewVideoRef}
                      autoPlay
                      playsInline
                      muted
                      className="w-full h-full object-cover rounded-lg"
                      style={{ transform: 'scaleX(-1)' }}
                    />
                    <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-50 text-white p-4">
                      <div className="flex justify-between items-center">
                        <div>
                          {previewPrediction ? (
                            <>
                              <div className="text-lg font-bold">
                                {previewPrediction.className}
                              </div>
                              <div className="text-sm">
                                신뢰도: {(previewPrediction.probability * 100).toFixed(1)}%
                              </div>
                            </>
                          ) : (
                            <div className="text-sm">분석 중...</div>
                          )}
                        </div>
                        <button
                          onClick={stopPreview}
                          className="bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600"
                        >
                          중지
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* 모델 내보내기 */}
            <div className="bg-white rounded-lg shadow">
              <div className="p-4 border-b">
                <h2 className="text-lg font-medium">모델 내보내기</h2>
              </div>
              <div className="p-4">
                <button
                  onClick={exportModel}
                  disabled={!model}
                  className={`w-full py-2 px-4 rounded-md ${
                    !model 
                      ? 'bg-gray-200 text-gray-400 cursor-not-allowed' 
                      : 'bg-green-600 text-white hover:bg-green-700'
                  }`}
                >
                  모델 내보내기
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* 웹캠 미리보기 */}
        {isWebcamActive && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
            <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full">
              <div className="p-4 border-b flex justify-between items-center">
                <h3 className="text-lg font-medium">
                  카메라 캡처: {classes[activeClassIndex]?.name}
                </h3>
                <button 
                  onClick={stopWebcam}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              <div className="p-4">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full rounded"
                />
                <div className="mt-4 flex justify-end">
                  <button
                    onClick={captureImage}
                    className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
                  >
                    캡처
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default TeachableMachine;