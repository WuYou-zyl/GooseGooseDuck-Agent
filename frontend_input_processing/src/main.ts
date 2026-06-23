import './assets/main.css'

import { createApp } from 'vue'
import App from './App.vue'
// 如果你不需要路由，可以暂时注释掉，但通常保留即可
import router from './router' 

const app = createApp(App)

app.use(router)

// 确保 App 挂载到 id 为 app of index.html 的元素上
app.mount('#app')