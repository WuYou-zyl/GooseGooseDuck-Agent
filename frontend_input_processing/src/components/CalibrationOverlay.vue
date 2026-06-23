<template>
  <div class="calibration-overlay" v-if="active" @mousedown="onMouseDown" @mousemove="onMouseMove" @mouseup="onMouseUp">
    <div class="instruction">
      步骤 {{ step + 1 }}/4: 请框选 <strong>{{ stepLabels[step] }}</strong>
    </div>
    
    <div v-if="isDrawing" class="drawing-box" :style="drawingBoxStyle"></div>
    
    <div v-for="(box, key) in results" :key="key" class="saved-box" :style="getBoxStyle(box)">
      {{ key }}
    </div>

    <div class="actions">
      <button @click="reset">重置</button>
      <button @click="cancel">取消</button>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue';
import { apiFetch } from '@/lib/api';

const props = defineProps(['active', 'canvasWidth', 'canvasHeight']);
const emit = defineEmits(['close']);

const step = ref(0);
const stepLabels = ['1号座位', '2号座位', '4号座位', '提示区'];
const keys = ['seat_1', 'seat_2', 'seat_4', 'speaker_ui'];
const results = ref({});

const isDrawing = ref(false);
const startPos = { x: 0, y: 0 };
const currentPos = { x: 0, y: 0 };

const onMouseDown = (e) => {
  isDrawing.value = true;
  const rect = e.currentTarget.getBoundingClientRect();
  startPos.x = e.clientX - rect.left;
  startPos.y = e.clientY - rect.top;
  currentPos.x = startPos.x;
  currentPos.y = startPos.y;
};

const onMouseMove = (e) => {
  if (!isDrawing.value) return;
  const rect = e.currentTarget.getBoundingClientRect();
  currentPos.x = e.clientX - rect.left;
  currentPos.y = e.clientY - rect.top;
};

const onMouseUp = async () => {
  if (!isDrawing.value) return;
  isDrawing.value = false;

  // 计算相对坐标 (x, y, w, h)
  const x = Math.min(startPos.x, currentPos.x);
  const y = Math.min(startPos.y, currentPos.y);
  const w = Math.abs(currentPos.x - startPos.x);
  const h = Math.abs(currentPos.y - startPos.y);

  results.value[keys[step.value]] = [Math.round(x), Math.round(y), Math.round(w), Math.round(h)];

  if (step.value < 3) {
    step.value++;
  } else {
    // 标定结束，提交到后端
    await apiFetch('/api/v1/calibrate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(results.value)
    });
    alert("标定完成！后端已重载坐标。");
    emit('close');
  }
};

const drawingBoxStyle = computed(() => ({
  left: Math.min(startPos.x, currentPos.x) + 'px',
  top: Math.min(startPos.y, currentPos.y) + 'px',
  width: Math.abs(currentPos.x - startPos.x) + 'px',
  height: Math.abs(currentPos.y - startPos.y) + 'px'
}));

const getBoxStyle = (box) => ({
  left: box[0] + 'px',
  top: box[1] + 'px',
  width: box[2] + 'px',
  height: box[3] + 'px'
});

const reset = () => { step.value = 0; results.value = {}; };
const cancel = () => emit('close');
</script>

<style scoped>
.calibration-overlay {
  position: absolute; top: 0; left: 0; width: 100%; height: 100%;
  background: rgba(0,0,0,0.3); cursor: crosshair; z-index: 100;
}
.instruction { position: absolute; top: 20px; left: 50%; transform: translateX(-50%); background: #fff; padding: 10px; border-radius: 4px; }
.drawing-box { position: absolute; border: 2px dashed #00ff00; background: rgba(0,255,0,0.1); }
.saved-box { position: absolute; border: 2px solid #ff0000; background: rgba(255,0,0,0.1); color: #fff; font-size: 10px; }
.actions { position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%); }
</style>
