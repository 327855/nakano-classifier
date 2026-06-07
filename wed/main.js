// ==================== 元素获取 ====================
// document.querySelector 根据 class / id / 标签名找到页面元素
const imageInput  = document.querySelector("#imageInput");
const predictBtn     = document.querySelector("#predictButton");
const resultArea  = document.querySelector("#resultArea");
const resultContent = document.querySelector("#resultContent");
const lightbox    = document.querySelector("#lightbox");
const lightboxImg = document.querySelector("#lightboxImg");
const lightboxCaption = document.querySelector("#lightboxCaption");
const closeLightbox = document.querySelector("#closeLightbox");


// ==================== 1. 卡片点击 → 弹出大图 ====================
// querySelectorAll 找到所有 .card，然后给每张卡片的 img 绑定点击事件
const cardImages = document.querySelectorAll(".card img");

cardImages.forEach((img) => {
    img.style.cursor = "pointer";          // 鼠标变成手型
    img.addEventListener("click", () => {
        // 点击后，把大图的 src 设为当前小图的 src
        lightboxImg.src = img.src;

        // 查找同卡片内的 h2 文字作为标题
        const card = img.closest(".card");
        const caption = card.querySelector("h2").innerText;
        lightboxCaption.innerText = caption;

        // 显示弹窗（添加 .show class）
        lightbox.classList.add("show");
    });
});


// ==================== 2. 关闭大图弹窗 ====================
// 点击 × 按钮关闭
closeLightbox.addEventListener("click", () => {
    lightbox.classList.remove("show");
});

// 点击弹窗背景也能关闭
lightbox.addEventListener("click", (e) => {
    if (e.target === lightbox) {
        lightbox.classList.remove("show");
    }
});


// ==================== 3. 识别按钮 → 调用后端 API ====================
// async 函数：标记为"异步函数"，内部可以用 await 等待结果
predictBtn.addEventListener("click", async () => {
    // 获取用户选择的文件
    const file = imageInput.files[0];

    // 没选文件时给出提示
    if (!file) {
        alert("请先选择一张图片！");
        return;
    }

    // 显示加载状态
    predictBtn.disabled = true;
    predictBtn.innerText = "识别中...";
    resultArea.style.display = "block";
    resultContent.innerHTML = "<p style='color:#888;'>正在识别，请稍候...</p>";

    try {
        // ==================== 读取图片并转为 base64 ====================
        // FileReader 将图片文件读取为 base64 字符串
        const reader = new FileReader();
        const imageBase64 = await new Promise((resolve, reject) => {
            reader.onload = () => resolve(reader.result);
            reader.onerror = () => reject(reader.error);
            reader.readAsDataURL(file);
        });

        // FileReader 产生的结果是 "data:image/xxx;base64,xxxxx" 格式
        // split(",")[1] 去掉前缀，只保留纯 base64 字符串部分
        const base64Data = imageBase64.split(",")[1];

        // ==================== 发送请求到后端 ====================
        // fetch 发送 POST 请求到 /api/predict
        // await 等待后端返回结果，代码暂停在这里，不会阻塞页面
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: base64Data }),
        });

        // 解析返回的 JSON
        const data = await response.json();

        if (data.error) {
            // 后端返回了错误
            resultContent.innerHTML = `<p style="color:red;">识别失败：${data.error}</p>`;
        } else {
            // 成功，显示结果
            resultContent.innerHTML = `
                <div class="result-card">
                    <div class="result-name">${data.name}</div>
                    <div class="result-confidence">置信度：${(data.confidence * 100).toFixed(1)}%</div>
                </div>
            `;
        }

    } catch (err) {
        // 网络请求失败（后端没启动、跨域等）
        resultContent.innerHTML = `<p style="color:red;">连接失败：后端服务可能未启动。</p>`;
    }

    // 恢复按钮状态
    predictBtn.disabled = false;
    predictBtn.innerText = "开始识别";
});
