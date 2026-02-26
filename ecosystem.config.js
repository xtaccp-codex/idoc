module.exports = {
  apps: [{
    name: "id-photo-api",
    // 指向虚拟环境中的 python 解释器
    interpreter: "./venv_id_photo/bin/python",
    script: "api_server.py",
    // 注意: pm2 的 cluster 模式只支持 NodeJS，Python 深度多进程必须使用 fork。
    // 如果要多进程，推荐增加多个 app 配置或采用 Gunicorn。这里回退为 fork
    exec_mode: "fork",
    autorestart: true,
    watch: false,
    // 2G 内存：总共跑两个进程，给每个进程 800M 左右的极限界限防炸服 (留给系统 400M)
    max_memory_restart: "800M",
    env: {
      NODE_ENV: "production",
    }
  }]
}
