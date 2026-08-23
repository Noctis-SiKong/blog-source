/* 把这段代码写到 custom.js 里，就能实现对应效果 */
document.addEventListener('DOMContentLoaded', function() {
  // 1. 点击头像弹出命运石之门经典台词（判空，避免无头像页面报错）
  var avatar = document.querySelector('#headerImg');
  if (avatar) {
    avatar.addEventListener('click', function() {
      alert('El Psy Kongroo！一切都是命运石之门的选择！');
    });
  }

  // 2. 加“回到顶部”按钮
  var backToTopBtn = document.createElement('button');
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
    display: none;
  `;
  document.body.appendChild(backToTopBtn);

  // 滚动页面时显示/隐藏按钮
  window.addEventListener('scroll', function() {
    if (window.scrollY > 300) {
      backToTopBtn.style.display = 'block';
    } else {
      backToTopBtn.style.display = 'none';
    }
  });

  // 点击按钮回到顶部
  backToTopBtn.addEventListener('click', function() {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
});
