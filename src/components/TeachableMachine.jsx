import React, { useState, useRef, useEffect, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';
import Card from './Card';
import Button from './Button';

// EPOCHS 상수를 컴포넌트 외부에서 선언
const EPOCHS = 20;

const TeachableMachine = () => {
  // 상태 관리
  const [classes, setClasses] = useState([
    { name: 'Class 1', samples: [] },
    { name: 'Class 2', samples: [] }
  ]);
  const [editingClassIndex, setEditingClassIndex] = useState(null);
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [activeClassIndex, setActiveClassIndex] = useState(null);
  const [model, setModel] = useState(null);
  const [baseModel, setBaseModel] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState({
    epoch: 0,
    loss: 0,
    accuracy: 0
  });
  const [prediction, setPrediction] = useState(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [trainSettings, setTrainSettings] = useState({
    epochs: EPOCHS,
    batchSize: 32,
    learningRate: 0.0001
  });
  const [trainingHistory, setTrainingHistory] = useState({
    loss: [],
    accuracy: []
  });

  // refs
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const fileInputRef = useRef(null);
  const webcamRef = useRef(null);
  const videoContainerRef = useRef(null);
  const predictionFrameId = useRef(null);

  // 웹캠 상태 관리를 위한 새로운 상태 추가
  const [isInitializing, setIsInitializing] = useState(false);

  // MobileNet 모델 로드
  useEffect(() => {
    // TensorFlow 백엔드 설정
    const setupTF = async () => {
      try {
        // CPU 백엔드로 폴백
        await tf.setBackend('cpu');
        console.log('TensorFlow 백엔드:', tf.getBackend());
        
        // MobileNet 로드
        const model = await mobilenet.load();
        setBaseModel(model);
        console.log('MobileNet 모델이 로드되었습니다.');
      } catch (error) {
        console.error('모델 로드 오류:', error);
      }
    };
    
    setupTF();

    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
    };
  }, []);

  // 웹캠 시작
  const startWebcam = useCallback(async (classIndex) => {
    try {
      // 기존 스트림 정리
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }

      // 먼저 상태 업데이트
      setIsWebcamActive(true);
      setActiveClassIndex(classIndex);

      // 웹캠 접근 요청
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false
      });

      // 상태가 업데이트된 후에 비디오 요소 설정
      await new Promise(resolve => requestAnimationFrame(resolve));

      if (!videoRef.current) {
        throw new Error('비디오 요소를 찾을 수 없습니다.');
      }

      // 비디오 설정
      videoRef.current.srcObject = stream;
      streamRef.current = stream;

      // 비디오 재생 시작
      try {
        await videoRef.current.play();
      } catch (playError) {
        console.error('비디오 재생 오류:', playError);
        throw playError;
      }
    } catch (error) {
      console.error('웹캠 접근 오류:', error);
      setIsWebcamActive(false);
      setActiveClassIndex(null);
      alert('웹캠을 시작할 수 없습니다: ' + error.message);
    }
  }, []);

  // 웹캠 중지
  const stopWebcam = () => {
    setIsWebcamActive(false);
    setPrediction(null);

    // 예측 루프 중지
    if (predictionFrameId.current) {
      cancelAnimationFrame(predictionFrameId.current);
      predictionFrameId.current = null;
    }

    // 스트림 정리
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => {
        track.stop();
      });
      streamRef.current = null;
    }

    // 비디오 요소 정리
    if (webcamRef.current) {
      webcamRef.current.srcObject = null;
    }
  };

  // 이미지 캡처
  const captureImage = () => {
    if (!videoRef.current || activeClassIndex === null) return;

    const video = videoRef.current;
    const canvas = document.createElement('canvas');
    
    // 비디오의 실제 크기 가져오기
    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;
    
    // 정사각형 크기 계산 (더 작은 쪽에 맞춤)
    const size = Math.min(videoWidth, videoHeight);
    
    // 캔버스 크기 설정 (224x224 대신 더 큰 해상도 사용)
    canvas.width = 448;  // 2배 크기
    canvas.height = 448;
    
    const ctx = canvas.getContext('2d');
    
    // 이미지 품질 개선을 위한 설정
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    
    // 비디오 중앙에서 정사각형 영역 캡처
    const sx = (videoWidth - size) / 2;
    const sy = (videoHeight - size) / 2;
    
    // 좌우 반전 적용
    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(
      video,
      sx, sy, size, size,  // 소스 영역
      -canvas.width, 0, canvas.width, canvas.height  // 대상 영역 (좌우 반전을 위해 x좌표 음수)
    );
    ctx.restore();
    
    // 캔버스의 이미지를 고품질 데이터 URL로 변환
    const dataUrl = canvas.toDataURL('image/jpeg', 0.95);  // 95% 품질
    
    // 클래스에 샘플 추가
    const updatedClasses = [...classes];
    updatedClasses[activeClassIndex].samples.push(dataUrl);
    setClasses(updatedClasses);
  };

  // 클래스 관리 함수들
  const addClass = () => {
    setClasses([...classes, { name: `Class ${classes.length + 1}`, samples: [] }]);
  };

  const removeClass = (index) => {
    setClasses(classes.filter((_, i) => i !== index));
  };

  const updateClassName = (index, newName) => {
    const updatedClasses = [...classes];
    updatedClasses[index].name = newName;
    setClasses(updatedClasses);
  };

  const removeSample = (classIndex, sampleIndex) => {
    const updatedClasses = [...classes];
    updatedClasses[classIndex].samples.splice(sampleIndex, 1);
    setClasses(updatedClasses);
  };

  const trainModel = async () => {
    if (!baseModel) return;

    try {
      setIsTraining(true);
      setTrainingStatus({ epoch: 0, loss: 0, accuracy: 0 });
      setTrainingHistory({ loss: [], accuracy: [] }); // 히스토리 초기화

      // 이미지 데이터 준비
      const processImages = classes.map((cls, classIndex) => 
        cls.samples.map(sample => new Promise((resolve) => {
          const img = new Image();
          img.onload = () => {
            const tensor = tf.tidy(() => {
              return tf.browser.fromPixels(img)
                .resizeNearestNeighbor([224, 224])
                .toFloat()
                .expandDims();
            });
            const activation = baseModel.infer(tensor, true);
            tensor.dispose();
            resolve({ activation, classIndex });
          };
          img.src = sample;
        }))
      ).flat();

      // 모든 이미지 처리 대기
      const results = await Promise.all(processImages);

      // 데이터셋 생성
      const xs = tf.concat(results.map(r => r.activation));
      const ys = tf.oneHot(
        tf.tensor1d(results.map(r => r.classIndex), 'int32'),
        classes.length
      );

      // 모델 생성 및 학습
      const newModel = tf.sequential({
        layers: [
          tf.layers.dense({
            inputShape: [1024],
            units: 128,
            activation: 'relu',
            kernelRegularizer: tf.regularizers.l2({ l2: 0.01 })
          }),
          tf.layers.dropout({ rate: 0.5 }),
          tf.layers.dense({
            units: classes.length,
            activation: 'softmax'
          })
        ]
      });

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
          onEpochEnd: (epoch, logs) => {
            const accuracy = Math.min((logs.acc * 100), 100);
            setTrainingStatus({
              epoch: epoch + 1,
              loss: logs.loss.toFixed(4),
              accuracy: accuracy.toFixed(2)
            });
            
            // 히스토리 업데이트
            setTrainingHistory(prev => ({
              loss: [...prev.loss, logs.loss],
              accuracy: [...prev.accuracy, accuracy]
            }));
          }
        }
      });

      setModel(newModel);

      // 메모리 정리
      xs.dispose();
      ys.dispose();

    } catch (error) {
      console.error('학습 오류:', error);
      alert('모델 학습 중 오류가 발생했습니다: ' + error.message);
    } finally {
      setIsTraining(false);
    }
  };

  const predictWebcam = async () => {
    if (!model || !baseModel) return;

    try {
      setIsWebcamActive(true);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();

        const predictFrame = async () => {
          if (!videoRef.current || !model) return;

          const tensor = tf.tidy(() => {
            return tf.browser.fromPixels(videoRef.current)
              .resizeNearestNeighbor([224, 224])
              .toFloat()
              .expandDims();
          });

          const activation = baseModel.infer(tensor, true);
          const predictions = model.predict(activation);
          const predictedIndex = predictions.argMax(1).dataSync()[0];
          setPrediction(classes[predictedIndex].name);

          tensor.dispose();
          activation.dispose();
          predictions.dispose();

          if (isWebcamActive) {
            requestAnimationFrame(predictFrame);
          }
        };

        predictFrame();
      }
    } catch (error) {
      console.error('웹캠 테스트 오류:', error);
      alert('웹캠 테스트를 시작할 수 없습니다: ' + error.message);
      setIsWebcamActive(false);
    }
  };

  const exportModel = async () => {
    if (!model) return;
    
    try {
      // 모델 구조와 가중치 저장
      await model.save('downloads://my-model');
      
      // 클래스 정보 저장
      const classInfo = classes.map(cls => ({
        name: cls.name
      }));
      
      // 클래스 정보를 JSON 파일로 저장
      const classInfoBlob = new Blob(
        [JSON.stringify(classInfo, null, 2)],
        { type: 'application/json' }
      );
      const classInfoUrl = URL.createObjectURL(classInfoBlob);
      
      // 다운로드 링크 생성 및 클릭
      const link = document.createElement('a');
      link.href = classInfoUrl;
      link.download = 'class-info.json';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(classInfoUrl);
      
      alert('모델과 클래스 정보가 저장되었습니다.');
    } catch (error) {
      console.error('모델 저장 오류:', error);
      alert('모델 저장 중 오류가 발생했습니다.');
    }
  };

  const handleImageUpload = (classIndex) => {
    const file = fileInputRef.current.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const updatedClasses = [...classes];
        updatedClasses[classIndex].samples.push(e.target.result);
        setClasses(updatedClasses);
      };
      reader.readAsDataURL(file);
      fileInputRef.current.value = ''; // 입력 초기화
    }
  };

  const stopTest = () => {
    setIsWebcamActive(false);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setPrediction(null);
  };

  // CSS 스타일을 추가
  const predictionOverlayStyle = {
    position: 'absolute',
    bottom: '10px',
    left: '10px',
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    color: 'white',
    padding: '8px 12px',
    borderRadius: '4px',
    fontSize: '14px'
  };

  // 웹캠 컨테이너 스타일
  const webcamContainerStyle = {
    position: 'relative',
    width: '100%',  // 컨테이너 너비를 100%로
    aspectRatio: '4/3',  // 4:3 비율 유지
    maxWidth: '400px',  // 최대 너비 제한
    margin: '0 auto',
    backgroundColor: 'black',
    overflow: 'hidden',
    borderRadius: '0.5rem'
  };

  // 웹캠 관련 코드 수정
  const setupWebcam = async () => {
    const videoElement = webcamRef.current;
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: 320, // 웹캠 해상도 조절
        height: 240
      }
    });
    videoElement.srcObject = stream;
    return new Promise((resolve) => {
      videoElement.onloadedmetadata = () => {
        resolve();
      };
    });
  };

  // predict 함수 수정
  const predict = async () => {
    if (!webcamRef.current || !model || !baseModel) return;

    try {
      const videoElement = webcamRef.current;
      
      // 비디오가 준비되지 않았다면 리턴
      if (!videoElement.videoWidth || !videoElement.videoHeight) {
        return;
      }

      const tensor = tf.tidy(() => {
        return tf.browser.fromPixels(videoElement)
          .resizeNearestNeighbor([224, 224])
          .toFloat()
          .expandDims();
      });

      const activation = baseModel.predict(tensor);
      const predictions = await model.predict(activation).data();
      
      const maxProbability = Math.max(...predictions);
      const predictedClassIndex = predictions.indexOf(maxProbability);
      
      setPrediction({
        className: classes[predictedClassIndex].name,
        probability: maxProbability
      });

      tensor.dispose();
      activation.dispose();
    } catch (error) {
      console.error('예측 중 오류:', error);
    }
  };

  // 웹캠 초기화 함수
  const initializeWebcam = async () => {
    if (!webcamRef.current) {
      console.error('비디오 요소가 없습니다');
      return false;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: { ideal: 640 },
          height: { ideal: 480 }
        }
      });

      webcamRef.current.srcObject = stream;
      streamRef.current = stream;

      return new Promise((resolve) => {
        webcamRef.current.onloadedmetadata = () => {
          webcamRef.current.play()
            .then(() => resolve(true))
            .catch(() => resolve(false));
        };
      });
    } catch (error) {
      console.error('웹캠 초기화 오류:', error);
      return false;
    }
  };

  // startPredicting 함수 수정
  const startPredicting = async () => {
    if (!model || !baseModel) {
      alert('먼저 모델을 학습시켜주세요.');
      return;
    }

    setIsInitializing(true);
    setPrediction(null);

    try {
      // 기존 웹캠 정리
      stopWebcam();

      // 웹캠 초기화
      const success = await initializeWebcam();
      if (!success) {
        throw new Error('웹캠을 초기화할 수 없습니다.');
      }

      setIsWebcamActive(true);

      // 예측 루프 시작 전 약간의 지연
      await new Promise(resolve => setTimeout(resolve, 1000));

      // 예측 시작
      const predictLoop = async () => {
        if (!isWebcamActive || !webcamRef.current) {
          return;
        }

        try {
          await predict();
          // 다음 프레임 예약 전에 상태 확인
          if (isWebcamActive && webcamRef.current) {
            predictionFrameId.current = requestAnimationFrame(predictLoop);
          }
        } catch (error) {
          console.error('예측 루프 오류:', error);
          if (error.message.includes('disposed')) {
            stopWebcam();
          }
        }
      };

      // 예측 루프 시작
      if (webcamRef.current && isWebcamActive) {
        predictLoop();
      }

    } catch (error) {
      console.error('웹캠 시작 오류:', error);
      stopWebcam();
      alert('웹캠을 시작할 수 없습니다: ' + error.message);
    } finally {
      setIsInitializing(false);
    }
  };

  // 파일 업로드 처리 함수 추가
  const handleFileUpload = async (classIndex, files) => {
    if (!files.length) return;

    try {
      const imagePromises = Array.from(files).map(file => {
        return new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = (e) => resolve(e.target.result);
          reader.onerror = (e) => reject(e);
          reader.readAsDataURL(file);
        });
      });

      const images = await Promise.all(imagePromises);
      
      setClasses(prevClasses => {
        const updatedClasses = [...prevClasses];
        updatedClasses[classIndex].samples.push(...images);
        return updatedClasses;
      });
    } catch (error) {
      console.error('파일 업로드 오류:', error);
      alert('이미지 업로드 중 오류가 발생했습니다.');
    }
  };

  // 그래프 컴포넌트 수정
  const Graph = ({ data, label, color }) => {
    if (!data || data.length === 0) return null;

    const maxY = Math.max(...data);
    const minY = Math.min(...data);
    const padding = 20;

    const getX = (i) => (i / (data.length - 1)) * (100 - padding * 2) + padding;
    const getY = (value) => {
      const normalizedValue = (value - minY) / (maxY - minY);
      return (1 - normalizedValue) * (100 - padding * 2) + padding;
    };

    // 선 그래프를 위한 path 생성
    const linePath = data.map((value, i) => {
      const x = getX(i);
      const y = getY(value);
      return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
    }).join(' ');

    return (
      <div className="h-40 bg-gray-50 rounded-lg p-2">
        <div className="w-full h-full relative">
          <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
            {/* 축 */}
            <line x1={padding} y1={padding} x2={padding} y2={100-padding} stroke="gray" strokeWidth="0.5" />
            <line x1={padding} y1={100-padding} x2={100-padding} y2={100-padding} stroke="gray" strokeWidth="0.5" />
            
            {/* 격자선 */}
            {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
              const y = padding + (100 - padding * 2) * ratio;
              return (
                <g key={ratio}>
                  <line 
                    x1={padding} 
                    y1={y} 
                    x2={100-padding} 
                    y2={y} 
                    stroke="gray" 
                    strokeWidth="0.1" 
                    strokeDasharray="2,2"
                  />
                  <text 
                    x={padding-5} 
                    y={y} 
                    fontSize="4" 
                    textAnchor="end" 
                    dominantBaseline="middle"
                  >
                    {((1 - ratio) * (maxY - minY) + minY).toFixed(2)}
                  </text>
                </g>
              );
            })}

            {/* 데이터 선 */}
            <path
              d={linePath}
              fill="none"
              stroke={color}
              strokeWidth="0.5"
            />
            
            {/* 데이터 포인트 */}
            {data.map((value, i) => (
              <circle
                key={i}
                cx={getX(i)}
                cy={getY(value)}
                r="1"
                fill={color}
              />
            ))}
          </svg>
          <div className="absolute top-0 left-2 text-xs text-gray-500">{label}</div>
        </div>
      </div>
    );
  };

  // 컴포넌트가 언마운트될 때 정리
  useEffect(() => {
    return () => {
      stopWebcam();
    };
  }, []);

  // TestCard 컴포넌트를 최상위 레벨로 이동
  const TestCard = () => (
    <div className="space-y-4">
      <div className="relative w-full aspect-video bg-black rounded-lg overflow-hidden">
        <video
          ref={webcamRef}
          playsInline
          muted
          className="w-full h-full object-cover"
          style={{ 
            transform: 'scaleX(-1)',
            display: isWebcamActive ? 'block' : 'none'
          }}
        />
        {prediction && isWebcamActive && (
          <div className="absolute bottom-4 left-4 bg-black bg-opacity-50 text-white p-2 rounded">
            {prediction.className}: {(prediction.probability * 100).toFixed(1)}%
          </div>
        )}
      </div>
      
      {!isWebcamActive ? (
        <Button
          onClick={startPredicting}
          className="w-full mb-4"
          disabled={!model || isInitializing}
        >
          {isInitializing ? '초기화 중...' : '테스트 시작'}
        </Button>
      ) : (
        <Button
          onClick={stopWebcam}
          className="w-full"
          variant="outline"
        >
          테스트 중지
        </Button>
      )}
    </div>
  );

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="flex flex-col sm:flex-row justify-between items-center mb-4 sm:mb-8">
        <div>
          <h2 className="text-base sm:text-lg font-semibold text-gray-600 mb-1 sm:mb-2">CodeWalks</h2>
          <h1 className="text-xl sm:text-2xl md:text-3xl font-bold">
            나의 데이터로 AI를 학습해요
          </h1>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-12 gap-4 sm:gap-8">
        <div className="md:col-span-1 lg:col-span-4 space-y-3 sm:space-y-4">
          {classes.map((cls, index) => (
            <Card key={index} className="p-3 sm:p-4">
              <div className="flex justify-between items-center mb-2 sm:mb-4">
                {editingClassIndex === index ? (
                  <input
                    type="text"
                    value={cls.name}
                    onChange={(e) => updateClassName(index, e.target.value)}
                    onBlur={() => setEditingClassIndex(null)}
                    autoFocus
                    className="border p-1 rounded"
                  />
                ) : (
                  <h3 
                    className="font-medium cursor-pointer"
                    onClick={() => setEditingClassIndex(index)}
                  >
                    {cls.name}
                  </h3>
                )}
                <Button
                  onClick={() => removeClass(index)}
                  variant="ghost"
                  className="text-red-500 hover:text-red-700"
                >
                  삭제
                </Button>
              </div>

              <div className="grid grid-cols-3 sm:grid-cols-4 gap-1 sm:gap-2">
                {cls.samples.map((sample, sampleIndex) => (
                  <div key={sampleIndex} className="relative aspect-square">
                    <img
                      src={sample}
                      alt={`Sample ${sampleIndex + 1}`}
                      className="w-full h-full object-cover rounded"
                    />
                    <button
                      onClick={() => removeSample(index, sampleIndex)}
                      className="absolute top-1 right-1 bg-red-500 text-white rounded-full w-5 h-5 flex items-center justify-center text-xs hover:bg-red-600"
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>

              <div className="flex flex-col sm:flex-row gap-2 mt-3 sm:mt-4">
                <Button
                  onClick={() => startWebcam(index)}
                  className="w-full sm:flex-1 text-xs sm:text-sm py-1.5 sm:py-2"
                  disabled={isWebcamActive}
                >
                  <div className="flex items-center justify-center space-x-1 sm:space-x-2">
                    <svg 
                      className="w-3 h-3 sm:w-4 sm:h-4" 
                      fill="none" 
                      stroke="currentColor" 
                      viewBox="0 0 24 24"
                    >
                      <path 
                        strokeLinecap="round" 
                        strokeLinejoin="round" 
                        strokeWidth={2} 
                        d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                      />
                    </svg>
                    <span className="whitespace-nowrap">웹캠으로 추가</span>
                  </div>
                </Button>

                <div className="w-full sm:flex-1">
                  <label className="block w-full">
                    <Button
                      className="w-full text-xs sm:text-sm py-1.5 sm:py-2"
                      variant="outline"
                      onClick={() => document.getElementById(`file-upload-${index}`).click()}
                    >
                      <div className="flex items-center justify-center space-x-1 sm:space-x-2">
                        <svg 
                          className="w-3 h-3 sm:w-4 sm:h-4" 
                          fill="none" 
                          stroke="currentColor" 
                          viewBox="0 0 24 24"
                        >
                          <path 
                            strokeLinecap="round" 
                            strokeLinejoin="round" 
                            strokeWidth={2} 
                            d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                          />
                        </svg>
                        <span className="whitespace-nowrap">이미지 업로드</span>
                      </div>
                    </Button>
                    <input
                      id={`file-upload-${index}`}
                      type="file"
                      accept="image/*"
                      multiple
                      className="hidden"
                      onChange={(e) => handleFileUpload(index, e.target.files)}
                    />
                  </label>
                </div>
              </div>
            </Card>
          ))}

          <Button
            variant="outline"
            onClick={addClass}
            className="w-full text-xs sm:text-sm py-1.5 sm:py-2"
          >
            + 새 클래스 추가
          </Button>
        </div>

        <div className="md:col-span-1 lg:col-span-8">
          {isWebcamActive ? (
            <Card className="p-4">
              <div className="relative aspect-video bg-black rounded-lg overflow-hidden mb-4">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="absolute inset-0 w-full h-full object-cover"
                  style={{ transform: 'scaleX(-1)' }}
                />
              </div>
              <div className="flex space-x-2">
                <Button onClick={captureImage} className="flex-1">
                  캡처
                </Button>
                <Button
                  onClick={() => {
                    setIsWebcamActive(false);
                    setActiveClassIndex(null);
                    if (streamRef.current) {
                      streamRef.current.getTracks().forEach(track => track.stop());
                      streamRef.current = null;
                    }
                  }}
                  variant="outline"
                  className="flex-1"
                >
                  닫기
                </Button>
              </div>
            </Card>
          ) : (
            <div className="grid grid-cols-2 gap-4 items-start">
              <Card className="p-4 h-fit">
                <h2 className="text-xl font-bold mb-4">모델 학습</h2>
                <p className="text-gray-600 mb-4">
                  각 클래스에 대한 샘플을 충분히 추가한 후 학습을 시작하세요.
                </p>

                <Button
                  onClick={trainModel}
                  className="w-full mb-4"
                  disabled={isTraining || classes.some(cls => cls.samples.length === 0)}
                >
                  {isTraining ? '학습 중...' : '학습 시작'}
                </Button>

                <button
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="flex items-center text-sm text-gray-600 hover:text-gray-800 mb-2"
                >
                  <svg
                    className={`w-4 h-4 mr-1 transform transition-transform ${
                      showAdvanced ? 'rotate-90' : ''
                    }`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 5l7 7-7 7"
                    />
                  </svg>
                  고급 설정
                </button>

                {showAdvanced && (
                  <div className="space-y-4 mb-4 p-4 bg-gray-50 rounded-lg">
                    <div>
                      <label className="block text-sm text-gray-700 mb-1">
                        에포크 수:
                      </label>
                      <input
                        type="number"
                        value={trainSettings.epochs}
                        onChange={(e) => setTrainSettings(prev => ({
                          ...prev,
                          epochs: Math.max(1, parseInt(e.target.value))
                        }))}
                        className="w-full px-3 py-2 border rounded-md"
                        min="1"
                      />
                    </div>
                    <div>
                      <label className="block text-sm text-gray-700 mb-1">
                        배치 크기:
                      </label>
                      <input
                        type="number"
                        value={trainSettings.batchSize}
                        onChange={(e) => setTrainSettings(prev => ({
                          ...prev,
                          batchSize: Math.max(1, parseInt(e.target.value))
                        }))}
                        className="w-full px-3 py-2 border rounded-md"
                        min="1"
                      />
                    </div>
                    <div>
                      <label className="block text-sm text-gray-700 mb-1">
                        학습률:
                      </label>
                      <input
                        type="number"
                        value={trainSettings.learningRate}
                        onChange={(e) => setTrainSettings(prev => ({
                          ...prev,
                          learningRate: parseFloat(e.target.value)
                        }))}
                        className="w-full px-3 py-2 border rounded-md"
                        step="0.0001"
                        min="0.0001"
                        max="1"
                      />
                    </div>
                  </div>
                )}

                {trainingStatus.epoch > 0 && (
                  <div className="mt-4 space-y-4">
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>진행 상태:</span>
                        <span className="font-medium">
                          {trainingStatus.epoch} / {trainSettings.epochs} 에포크
                        </span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span>손실:</span>
                        <span className="font-medium">{trainingStatus.loss}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span>정확도:</span>
                        <span className="font-medium">{trainingStatus.accuracy}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-blue-600 h-2 rounded-full transition-all"
                          style={{ 
                            width: `${Math.min((trainingStatus.epoch / trainSettings.epochs) * 100, 100)}%` 
                          }}
                        />
                      </div>
                    </div>

                    <div className="space-y-4">
                      <Graph 
                        data={trainingHistory.loss} 
                        label="Loss" 
                        color="#ef4444"  // red-500
                      />
                      <Graph 
                        data={trainingHistory.accuracy} 
                        label="Accuracy" 
                        color="#3b82f6"  // blue-500
                      />
                    </div>
                  </div>
                )}
              </Card>

              <Card className="p-4 h-fit">
                <h2 className="text-xl font-bold mb-4">모델 테스트</h2>
                <p className="text-gray-600 mb-4">
                  학습된 모델을 테스트하거나 내보내세요.
                </p>
                <TestCard />

                <div className="border-t border-gray-200 my-4"></div>

                <Button
                  onClick={exportModel}
                  className="w-full"
                  disabled={!model}
                  variant="outline"
                >
                  <div className="flex items-center justify-center space-x-2">
                    <svg 
                      className="w-4 h-4" 
                      fill="none" 
                      stroke="currentColor" 
                      viewBox="0 0 24 24"
                    >
                      <path 
                        strokeLinecap="round" 
                        strokeLinejoin="round" 
                        strokeWidth={2} 
                        d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                      />
                    </svg>
                    <span>모델 내보내기</span>
                  </div>
                </Button>
              </Card>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default TeachableMachine;