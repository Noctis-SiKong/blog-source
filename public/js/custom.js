/* 把这段代码写到 custom.js 里，就能实现对应效果 */
// 1. 点击头像弹出命运石之门经典台词
document.querySelector('.avatar-img').addEventListener('click', function() {
  alert('El Psy Kongroo！一切都是命运石之门的选择！');
});

// 2. 加“回到顶部”按钮
// 第一步：创建按钮元素
const backToTopBtn = document.createElement('button');
backToTopBtn.innerText = '回到顶部';
backToTopBtn.style.cssText = `
  position: fixed;
  bottom: 20px;
  right: 20px;
  padding: 10px;
  background: #D62828;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  display: none; /* 默认隐藏 */
`;
document.body.appendChild(backToTopBtn);

// 第二步：滚动页面时显示/隐藏按钮
window.addEventListener('scroll', function() {
  if (window.scrollY > 300) { // 滚动超过300px显示
    backToTopBtn.style.display = 'block';
  } else {
    backToTopBtn.style.display = 'none';
  }
});

// 第三步：点击按钮回到顶部
backToTopBtn.addEventListener('click', function() {
  window.scrollTo({ top: 0, behavior: 'smooth' }); // 平滑滚动
});