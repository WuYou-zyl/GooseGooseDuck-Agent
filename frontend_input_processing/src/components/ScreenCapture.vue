<template>
  <div class="vision-section">
    <div class="controls-header">
      <div v-if="currentActiveSeat" class="active-indicator">
        {{ currentActiveSeat }}号发言中
      </div>
      <div class="spacer"></div>
      <div class="btn-group">
        <div class="zoom-tag">{{ Math.round(zoomLevel * 100) }}%</div>
        <button @click="changeZoom(-0.1)" class="btn-tool">-</button>
        <button @click="changeZoom(0.1)" class="btn-tool">+</button>
        <button @click="toggleCalibration" :class="{ 'btn-warn': isCalibrating }" class="btn-tool">
          {{ isCalibrating ? '取消' : '标定' }}
        </button>
        <button @click="requestNewFrame" class="btn-tool">刷新</button>
        <label class="toggle-btn">
            <input type="checkbox" v-model="showGrid"> 显示网格
        </label>
      </div>
    </div>

    <div class="canvas-viewport">
        <div class="canvas-relative-wrapper" :style="containerStyle">
        <canvas ref="canvasRef" :width="originalSize.w" :height="originalSize.h"></canvas>

        <div v-if="showGrid" class="grid-preview-layer">
            <div v-for="(roi, index) in previewRois.seats" :key="'seat-'+index" 
                class="roi-box" :style="getRoiStyle(roi)">
            <span class="roi-tag">Seat {{ index + 1 }}</span>
            </div>
            <div v-if="previewRois.speaker_ui" 
                class="roi-box ui-style" :style="getRoiStyle(previewRois.speaker_ui)">
            <span class="roi-tag">Speaker UI</span>
            </div>
        </div>

        <div v-if="isCalibrating" 
            class="calibration-layer"
            @mousedown="onMouseDown" 
            @mousemove="onMouseMove" 
            @mouseup="onMouseUp">
            
            <div class="guide-tip">正在标定: {{ stepLabels[calStep] }} (鼠标拖拽框选)</div>
            
            <div v-if="isDrawing" class="drawing-rect" :style="drawingRectStyle"></div>
            
            <div v-if="hasDrawn && tempBox" class="temp-rect" :style="getRoiStyle(tempBox)">
            <div class="confirm-actions-floating">
                <button @click.stop="confirmStep" class="btn-confirm-mini">确认</button>
                <button @click.stop="retryStep" class="btn-retry-mini">重画</button>
            </div>
            </div>
        </div>
        </div>

    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, inject, type CSSProperties } from 'vue';
import { apiFetch } from '@/lib/api';

// 1. 声明 emit
const emit = defineEmits(['update-size']);

// --- 基础配置 ---
const stepLabels = ['1号位', '2号位', '4号位', '提示区'];
const keys = ['seat_1', 'seat_2', 'seat_4', 'speaker_ui'] as const;

// 注入 App.vue 提供的 WebSocket 上下文
const wsContext = inject('ws_context') as any;

// --- 状态定义 ---
const showImage = ref(true);
const isCalibrating = ref(false);
const currentActiveSeat = ref<number | null>(null);
const zoomLevel = ref(0.6);

const canvasRef = ref<HTMLCanvasElement | null>(null);
const originalSize = reactive({ w: 800, h: 450 });
const bgImage = new Image();
const showGrid = ref(true);

// 标定逻辑状态
const calStep = ref(0);
const isDrawing = ref(false);
const hasDrawn = ref(false);
const tempBox = ref<number[] | null>(null);
const startPos = reactive({ x: 0, y: 0 });
const currentPos = reactive({ x: 0, y: 0 });
const results = ref<Record<string, number[]>>({});
const previewRois = ref<{ seats?: number[][], speaker_ui?: number[] }>({});

// --- 计算属性 ---
const containerStyle = computed(() => ({
  width: `${originalSize.w * zoomLevel.value}px`,
  height: `${originalSize.h * zoomLevel.value}px`,
  position: 'relative' as const,
  transition: 'all 0.2s ease-out'
}));

const drawingRectStyle = computed((): CSSProperties => {
  const x = Math.min(startPos.x, currentPos.x) / originalSize.w * 100;
  const y = Math.min(startPos.y, currentPos.y) / originalSize.h * 100;
  const w = Math.abs(currentPos.x - startPos.x) / originalSize.w * 100;
  const h = Math.abs(currentPos.y - startPos.y) / originalSize.h * 100;
  return { position: 'absolute' as const, left: `${x}%`, top: `${y}%`, width: `${w}%`, height: `${h}%` };
});

// 封装一个尺寸上报函数
const emitSizeChange = () => {
  emit('update-size', {
    width: originalSize.w * zoomLevel.value,
    height: originalSize.h * zoomLevel.value
  });
};

// --- 方法：响应父组件数据 ---
const handleVisionData = (data: any) => {
  if (data.type === 'processed_frame') {
    currentActiveSeat.value = data.active_seat;
  }
  if (data.type === 'calibration_frame') {
    bgImage.onload = () => {
      originalSize.w = bgImage.naturalWidth;
      originalSize.h = bgImage.naturalHeight;
      renderCanvas();
      
      // 2. 视频帧加载后，通知父组件当前的视觉宽度和高度
      emitSizeChange();
    };
    bgImage.src = data.image;
  }
};

const renderCanvas = () => {
  const ctx = canvasRef.value?.getContext('2d');
  if (ctx && showImage.value) {
    ctx.clearRect(0, 0, originalSize.w, originalSize.h);
    ctx.drawImage(bgImage, 0, 0);
  }
};

// --- 方法：指令发送 ---
const requestNewFrame = () => {
  wsContext.sendMessage({ type: 'request_capture_frame' });
};

const fetchCurrentConfig = async () => {
  try {
    const res = await apiFetch("/api/v1/calibration/current");
    const data = await res.json();
    if (data.config_preview) previewRois.value = data.config_preview;
  } catch (e) { console.error("加载配置失败", e); }
};

// 3. 缩放改变时也要重新上报
const changeZoom = (delta: number) => {
  const newZoom = zoomLevel.value + delta;
  zoomLevel.value = Math.min(Math.max(0.3, newZoom), 3.0);
  emitSizeChange();
};

const onShowImageChange = () => { 
  if (showImage.value) requestNewFrame(); 
  else isCalibrating.value = false; 
};

const toggleCalibration = () => {
  isCalibrating.value = !isCalibrating.value;
  if (isCalibrating.value) {
      requestNewFrame(); // 标定时刷新一帧背景
      resetCalibState();
  }
};

const resetCalibState = () => {
  calStep.value = 0;
  results.value = {};
  hasDrawn.value = false;
  tempBox.value = null;
};

// --- 标定绘制逻辑 ---
const onMouseDown = (e: MouseEvent) => {
  if (hasDrawn.value) return;
  isDrawing.value = true;
  const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
  const scaleX = originalSize.w / rect.width;
  const scaleY = originalSize.h / rect.height;
  startPos.x = (e.clientX - rect.left) * scaleX;
  startPos.y = (e.clientY - rect.top) * scaleY;
  currentPos.x = startPos.x;
  currentPos.y = startPos.y;
};

const onMouseMove = (e: MouseEvent) => {
  if (!isDrawing.value) return;
  const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
  currentPos.x = (e.clientX - rect.left) * (originalSize.w / rect.width);
  currentPos.y = (e.clientY - rect.top) * (originalSize.h / rect.height);
};

const onMouseUp = () => {
  if (!isDrawing.value) return;
  isDrawing.value = false;
  const x = Math.min(startPos.x, currentPos.x);
  const y = Math.min(startPos.y, currentPos.y);
  const w = Math.abs(currentPos.x - startPos.x);
  const h = Math.abs(currentPos.y - startPos.y);
  if (w > 5 && h > 5) {
    tempBox.value = [Math.round(x), Math.round(y), Math.round(w), Math.round(h)];
    hasDrawn.value = true;
  }
};

// 修改 confirmStep 逻辑以匹配 keys = ['seat_1', 'seat_2', 'seat_4', 'speaker_ui']
const confirmStep = async () => {
  const currentKey = keys[calStep.value];

  // 1. 增加非空校验，确保 key 存在
  // 2. 使用类型断言 (as string) 解决索引签名问题
  if (currentKey && tempBox.value) {
    (results.value as Record<string, number[]>)[currentKey] = [...tempBox.value];
  }
  
  hasDrawn.value = false;
  tempBox.value = null;

  // 这里的 keys.length 目前是 4 (1, 2, 4号位 + UI区)
  if (calStep.value < keys.length - 1) {
    calStep.value++;
  } else {
    await submitCalibration();
  }
};

const retryStep = () => { hasDrawn.value = false; tempBox.value = null; };

const submitCalibration = async () => {
  try {
    const res = await apiFetch("/api/v1/calibrate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(results.value)
    });
    const data = await res.json();
    if (data.status === 'success') {
      previewRois.value = data.config_preview;
      isCalibrating.value = false;
    }
  } catch (e) { alert("上传标定配置失败"); }
};

// 核心：修正 getRoiStyle 支持 tempBox 数组 [x, y, w, h]
const getRoiStyle = (roi: any): CSSProperties => {
  if (!roi) return {};
  // 兼容后端对象格式 {x,y,w,h} 和前端临时数组格式 [x,y,w,h]
  const [x, y, w, h] = Array.isArray(roi) ? roi : [roi.x, roi.y, roi.w, roi.h];
  
  return {
    position: 'absolute',
    left: `${(x / originalSize.w) * 100}%`,
    top: `${(y / originalSize.h) * 100}%`,
    width: `${(w / originalSize.w) * 100}%`,
    height: `${(h / originalSize.h) * 100}%`,
  };
};

// 暴露接口给 App.vue
defineExpose({ handleVisionData, requestNewFrame });

// 初始化加载配置
fetchCurrentConfig();
</script>

<style scoped>
.controls { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }

.zoom-controls {
  display: flex;
  align-items: center;
  gap: 6px;
  background: #2a2a2a;
  padding: 4px 8px;
  border-radius: 6px;
}
.zoom-text { font-family: monospace; min-width: 40px; text-align: center; color: #42b883; font-size: 12px; }
.btn-mini { background: #444; color: white; border: none; padding: 2px 6px; cursor: pointer; border-radius: 3px; font-size: 11px; }

.canvas-container { background: #000; position: relative; flex-shrink: 0; box-shadow: 0 0 20px rgba(0,0,0,0.5); }
canvas { width: 100%; height: auto; display: block; }
.hidden-canvas { visibility: hidden; height: 0; }

.calibration-layer, .grid-preview-layer { position: absolute; inset: 0; pointer-events: none; }
.calibration-layer { pointer-events: all; background: rgba(0,0,0,0.4); cursor: crosshair; z-index: 10; }

.roi-box { border: 1px solid rgba(66, 184, 131, 0.4); background: rgba(66, 184, 131, 0.05); transition: all 0.3s; }
.roi-box.active { border-color: #00ff00; background: rgba(0, 255, 0, 0.15); box-shadow: inset 0 0 10px rgba(0, 255, 0, 0.2); }
.roi-tag { position: absolute; top: -18px; left: 0; background: #42b883; color: white; font-size: 10px; padding: 0 4px; border-radius: 2px; }

.drawing-rect { border: 2px dashed #42b883; background: rgba(66, 184, 131, 0.1); pointer-events: none;}
.temp-rect { position: absolute; border: 2px solid #f39c12; background: rgba(243, 156, 18, 0.2); pointer-events: none; }

.guide-tip { 
    position: absolute; top: 15px; left: 50%; transform: translateX(-50%); 
    background: #42b883; padding: 6px 15px; border-radius: 20px; 
    font-weight: bold; font-size: 12px; z-index: 20; color: #000;
}

.confirm-actions-floating {
  position: absolute; bottom: -40px; right: 0; display: flex; gap: 5px; pointer-events: all; z-index: 100;
}

.btn-confirm-mini { background: #42b883; color: #000; border: none; padding: 3px 8px; border-radius: 4px; cursor: pointer; font-size: 11px; font-weight: bold;}
.btn-retry-mini { background: #e74c3c; color: white; border: none; padding: 3px 8px; border-radius: 4px; cursor: pointer; font-size: 11px; font-weight: bold;}

.toggle-btn { display: flex; align-items: center; gap: 5px; font-size: 12px; cursor: pointer; color: #aaa; }
.btn-warning { background: #f39c12; color: #000; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 12px; font-weight: bold; }
.btn-secondary { background: #444; color: #eee; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 12px; }

.status-badge { background: rgba(0, 255, 0, 0.1); color: #00ff00; border: 1px solid #00ff00; padding: 3px 10px; border-radius: 4px; font-size: 12px; margin-left: auto; }
.pulse { animation: blink 2s infinite; }
.ui-style { border: 1px dashed #f39c12 !important; background: rgba(243, 156, 18, 0.05) !important; }

/* ScreenCapture.vue */


.actions-group {
  display: flex;
  align-items: center;
  gap: 8px;
}

.btn-mini-rect {
  background: #333;
  color: #eee;
  border: 1px solid #444;
  padding: 2px 8px;
  font-size: 11px;
  border-radius: 4px;
  cursor: pointer;
}

.viewport-container {
  transition: all 0.3s ease;
  display: flex;
  justify-content: center;
  align-items: flex-start;
  padding-top: 10px;
}

.viewport-container.minimized {
  height: 150px;
  opacity: 0.6;
}
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
</style>

<style scoped>
.vision-section {
  padding: 8px;
  display: flex;
  flex-direction: column;
}

.controls-header {
  display: flex;
  align-items: center;
  margin-bottom: 6px;
}

.spacer { flex: 1; }

.btn-group {
  display: flex;
  gap: 4px;
  align-items: center;
}

.btn-tool {
  background: #333;
  border: 1px solid #444;
  color: #eee;
  padding: 2px 6px;
  font-size: 11px;
  border-radius: 3px;
  cursor: pointer;
}

.btn-tool:hover { background: #444; }
.btn-warn { color: #f39c12; border-color: #f39c12; }

.zoom-tag { font-size: 10px; color: #666; margin-right: 4px; }

.canvas-viewport {
  overflow: hidden;
  border-radius: 4px;
  background: #000;
}

.active-indicator {
  font-size: 11px;
  color: #42b883;
  background: rgba(66, 184, 131, 0.1);
  padding: 2px 8px;
  border-radius: 10px;
}
</style>
