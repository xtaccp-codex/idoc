module.exports = {
  apps: [{
    name: "id-photo-api",
    // 指向虚拟环境中的 python 解释器
    interpreter: "./venv_id_photo/bin/python",
    script: "api_server.py",
    // 定时重启
    // 分钟[0-59] 小时[0-23] 天[1-31] 月份[1-12] 星期数[0-6]
    cron_restart: '0 4 * * *',
    // 注意: pm2 的 cluster 模式只支持 NodeJS，Python 深度多进程必须使用 fork。
    // 如果要多进程，推荐增加多个 app 配置或采用 Gunicorn。这里回退为 fork
    exec_mode: "fork",
    autorestart: true,
    watch: false,
    // 2G 内存：AI 推理峰值约 800M~1G，设 1500M 留足空间防止每次请求后都被 PM2 误杀重启
    max_memory_restart: "1500M",
    env: {
      NODE_ENV: "production",
    }
  }]
}
