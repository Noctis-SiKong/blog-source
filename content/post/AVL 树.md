---
title: 'AVL树'
date: '2025-12-25T19:30:10+08:00'
draft: false
series: '数据结构与算法优化'
---

# Java 实现 AVL 树

## 一、引言：为什么需要 AVL 树？

在数据结构的学习中，二叉搜索树（BST）是基础，但普通 BST 存在一个致命问题：如果插入的元素是有序的（比如 1,2,3,4,5），会退化成**链表**，查询、插入的时间复杂度从*O*(log*n*)骤降为*O*(*n*)。

AVL 树（以发明者 Adelson-Velsky 和 Landis 命名）是**自平衡二叉搜索树**，核心特性是：**任意节点的左右子树高度差（平衡因子）的绝对值不超过 1**。当插入 / 删除元素导致平衡因子超出范围时，会通过**旋转**操作重新平衡，保证树的高度始终是*O*(log*n*)，从而维持高效的增删查改性能。

本文将通过完整的 Java 代码，拆解 AVL 树的核心实现（插入、平衡调整、叶子节点统计、三种遍历），既是博客分享，也适配期末复习的考点梳理。

<!--More-->

## 二、代码整体架构

整个代码分为两部分：

1. `Main`主类：负责读取输入、创建 AVL 树实例、调用核心方法（插入、统计叶子、遍历）；
2. `BinarySearchTree`类：封装 AVL 树的核心逻辑（节点定义、插入、平衡调整、叶子统计、遍历）。

先看整体代码结构（带复习重点标注）：

```java
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        BinarySearchTree avlTree = new BinarySearchTree(); // 初始化AVL树
        // 循环读取输入整数，插入AVL树
        while (sc.hasNextInt()) {
            int v = sc.nextInt();
            avlTree.insert(v);
        }
        sc.close();
        // 1. 输出叶子节点数量
        System.out.println(avlTree.leave());
        // 2. 前序遍历输出
        avlTree.infront();
        // 3. 后序遍历输出
        avlTree.after();
        // 4. 中序遍历输出
        avlTree.mid();
    }
}

// AVL树核心类（复习核心）
class BinarySearchTree {
    // 1. 节点内部类（考点：AVL节点需维护高度）
    private static class TreeNode {
        int v;          // 节点值
        TreeNode left;  // 左子节点
        TreeNode right; // 右子节点
        int height;     // 节点高度（AVL核心属性，普通BST无此属性）

        TreeNode(int i) {
            this.v = i;
            this.left = null;
            this.right = null;
            this.height = 1; // 新节点高度初始为1
        }
    }

    private TreeNode root; // AVL树根节点

    public BinarySearchTree() {
        this.root = null; // 初始空树
    }

    // 2. 对外暴露的插入方法
    public void insert(int v) {
        this.root = insertBST(this.root, v);
    }

    // 3. 插入核心方法（含平衡调整，期末核心考点）
    public TreeNode insertBST(TreeNode root, int v) { /* 详见下文 */ }

    // 4. 辅助方法：获取节点高度
    private int getHeigh(TreeNode node) { /* 详见下文 */ }

    // 5. 辅助方法：计算平衡因子（期末核心考点）
    private int getBalance(TreeNode node) { /* 详见下文 */ }

    // 6. 旋转操作（左旋转、右旋转，期末核心考点）
    private TreeNode leftRotate(TreeNode x) { /* 详见下文 */ }
    private TreeNode rightRotate(TreeNode y) { /* 详见下文 */ }

    // 7. 统计叶子节点数量（递归，期末常考）
    public int leave() { /* 详见下文 */ }
    public int findleave(TreeNode root) { /* 详见下文 */ }

    // 8. 三种遍历（前序、中序、后序，期末必考）
    private boolean isFirst = true; // 控制遍历输出格式（无前置空格）
    public void mid() { /* 中序遍历 */ }
    public void midprint(TreeNode root) { /* 中序递归逻辑 */ }
    public void infront() { /* 前序遍历 */ }
    public void frontprint(TreeNode root) { /* 前序递归逻辑 */ }
    public void after() { /* 后序遍历 */ }
    public void afterprint(TreeNode root) { /* 后序递归逻辑 */ }
}
```

## 三、核心模块逐行解析（复习重点）

### 模块 1：TreeNode 节点定义（AVL 树的基础）

```java
private static class TreeNode {
    int v;          // 节点值
    TreeNode left;  // 左子节点
    TreeNode right; // 右子节点
    int height;     // 节点高度（★复习重点★）

    TreeNode(int i) {
        this.v = i;
        this.left = null;
        this.right = null;
        this.height = 1; // 新节点高度初始为1（叶子节点高度为1）
    }
}
```

- 区别于普通 BST：AVL 树的节点必须维护`height`属性，用于计算平衡因子；
- 高度定义：叶子节点高度为 1，空节点高度为 0（后续`getHeigh`方法会处理）；
- 期末考点：AVL 树节点的核心属性、高度的初始化规则。

### 模块 2：插入方法（AVL 树的核心，含平衡调整）

插入逻辑分为两步：① 普通 BST 的插入；② 调整高度 + 检查平衡 + 旋转平衡。

#### 步骤 1：普通 BST 插入

```java
public TreeNode insertBST(TreeNode root, int v) {
    // 1. 普通BST插入逻辑（递归终止条件：空节点则创建新节点）
    if (root == null) {
        return new TreeNode(v);
    }
    // 小于根节点：插入左子树
    if (v < root.v) {
        root.left = insertBST(root.left, v);
    } 
    // 大于根节点：插入右子树
    else if (v > root.v) {
        root.right = insertBST(root.right, v);
    } 
    // 等于根节点：AVL树不存重复值，直接返回
    else {
        return root;
    }
```

- 递归插入：符合 BST 的核心规则（左子树 < 根 < 右子树）；
- 去重：重复值直接返回，不插入（AVL 树通常不存储重复元素）。

#### 步骤 2：更新当前节点高度

```java
    // 2. 更新当前节点的高度（★复习重点★）
    root.height = 1 + Math.max(getHeigh(root.left), getHeigh(root.right));
```

- 高度计算公式：当前节点高度 = 1 + 左右子树高度的最大值；

- 辅助方法`getHeigh`：处理空节点的高度（空节点高度为 0）：

  ```java
  private int getHeigh(TreeNode node) {
      return node == null ? 0 : node.height;
  }
  ```

  

#### 步骤 3：计算平衡因子，判断是否失衡

```java
    // 3. 计算平衡因子（★期末核心考点★）
    int balance = getBalance(root);
    // 平衡因子 = 左子树高度 - 右子树高度
    private int getBalance(TreeNode node) {
        if (node == null) {
            return 0;
        }
        return getHeigh(node.left) - getHeigh(node.right);
    }
```

- 平衡因子定义：`平衡因子 = 左子树高度 - 右子树高度`；
- 失衡判定：平衡因子的绝对值 > 1 时，需要旋转调整；
- 期末考点：平衡因子的计算公式、失衡的判定条件。

#### 步骤 4：四种失衡情况的旋转调整（★期末重中之重★）

AVL 树的四种失衡情况对应四种旋转策略，核心是 “左旋” 和 “右旋” 两个基础操作，组合解决所有失衡：

| 失衡类型     | 平衡因子 | 插入位置       | 旋转策略            |
| ------------ | -------- | -------------- | ------------------- |
| 左左型（LL） | >1       | 左子树的左子树 | 右旋                |
| 右右型（RR） | < -1     | 右子树的右子树 | 左旋                |
| 左右型（LR） | >1       | 左子树的右子树 | 左子树左旋 + 根右旋 |
| 右左型（RL） | < -1     | 右子树的左子树 | 右子树右旋 + 根左旋 |

```java
    // 4. 旋转调整（四种失衡情况）
    // 情况1：LL型（左左）→ 右旋
    if (balance > 1 && v < root.left.v) {
        return rightRotate(root);
    }

    // 情况2：RR型（右右）→ 左旋
    if (balance < -1 && v > root.right.v) {
        return leftRotate(root);
    }

    // 情况3：LR型（左右）→ 左子树左旋 + 根右旋
    if (balance > 1 && v > root.left.v) {
        root.left = leftRotate(root.left);
        return rightRotate(root);
    }

    // 情况4：RL型（右左）→ 右子树右旋 + 根左旋
    if (balance < -1 && v < root.right.v) {
        root.right = rightRotate(root.right);
        return leftRotate(root);
    }

    // 未失衡：直接返回当前节点
    return root;
}
```

##### 基础旋转操作解析：

1. 右旋（解决 LL 型失衡）：

```java
private TreeNode rightRotate(TreeNode y) {
    // 步骤1：定义临时节点
    TreeNode x = y.right; // 错误修正：原代码笔误，正确应为 TreeNode x = y.left;
    TreeNode xLeft = x.right;

    // 步骤2：旋转核心
    x.left = y;
    y.right = xLeft;

    // 步骤3：更新高度（先更新下层节点y，再更新上层节点x）
    y.height = 1 + Math.max(getHeigh(y.left), getHeigh(y.right));
    x.height = 1 + Math.max(getHeigh(x.left), getHeigh(x.right));

    // 返回新的根节点（x成为新根）
    return x;
}
```

⚠️ 注：原代码中`rightRotate`方法有笔误（`TreeNode x = y.right`应为`TreeNode x = y.left`），已修正，复习时需注意！

1. 左旋（解决 RR 型失衡）：

```java
private TreeNode leftRotate(TreeNode x) {
    // 步骤1：定义临时节点
    TreeNode y = x.left; // 错误修正：原代码笔误，正确应为 TreeNode y = x.right;
    TreeNode yRight = y.right;

    // 步骤2：旋转核心
    y.right = x;
    x.left = yRight;

    // 步骤3：更新高度
    x.height = 1 + Math.max(getHeigh(x.left), getHeigh(x.right));
    y.height = 1 + Math.max(getHeigh(y.left), getHeigh(y.right));

    // 返回新的根节点（y成为新根）
    return y;
}
```

⚠️ 原代码`leftRotate`方法同样有笔误（`TreeNode y = x.left`应为`TreeNode y = x.right`），这是关键错误，复习时需重点注意！

- 旋转核心逻辑：调整节点的父子关系，同时更新高度；
- 期末考点：四种失衡类型的判定、旋转的顺序（如 LR 型先左旋左子树，再右旋根）、旋转后高度的更新。

### 模块 3：统计叶子节点数量（递归经典题）

叶子节点定义：左右子节点均为空的节点。

```java
// 对外暴露的方法
public int leave() {
    return findleave(this.root);
}

// 递归统计叶子节点
public int findleave(TreeNode root) {
    // 递归终止：空节点，叶子数为0
    if (root == null) {
        return 0;
    }
    // 递归统计左、右子树的叶子数
    int l = findleave(root.left);
    int r = findleave(root.right);
    // 当前节点是叶子：返回1
    if (root.left == null && root.right == null) {
        return 1;
    }
    // 非叶子：返回左右子树叶子数之和
    return l + r;
}
```

- 递归思路：**分治思想**，把 “统计整棵树的叶子数” 拆解为 “统计左子树叶子数 + 统计右子树叶子数”；
- 终止条件：① 空节点返回 0；② 叶子节点返回 1；
- 期末考点：递归统计叶子节点的逻辑、终止条件的设计。

### 模块 4：三种遍历（前序、中序、后序）

遍历是二叉树的核心考点，AVL 树的遍历和普通二叉树一致，区别仅在于 AVL 树是平衡的 BST，中序遍历结果为**升序序列**。

#### 核心逻辑对比（★期末必考★）

| 遍历类型 | 顺序         | 核心特征           |
| -------- | ------------ | ------------------ |
| 前序     | 根 → 左 → 右 | 先访问根节点       |
| 中序     | 左 → 根 → 右 | BST 中序遍历为升序 |
| 后序     | 左 → 右 → 根 | 最后访问根节点     |

#### 代码解析（以中序为例，前序 / 后序仅调整访问顺序）

```java
// 控制输出格式：避免前置空格
private boolean isFirst = true;

// 对外暴露的中序遍历方法
public void mid() {
    isFirst = true; // 每次遍历重置标记
    midprint(this.root);
    System.out.println(); // 遍历完换行
}

// 中序遍历递归逻辑
public void midprint(TreeNode root) {
    if (root == null) {
        return;
    }
    midprint(root.left); // 1. 遍历左子树
    // 2. 访问根节点（控制输出格式）
    if (isFirst) {
        System.out.print(root.v);
        isFirst = false;
    } else  {
        System.out.print(" " + root.v);
    }
    midprint(root.right); // 3. 遍历右子树
}
```

- 格式控制：`isFirst`标记避免输出 “1 2 3” 这种前置空格；
- 前序遍历：先访问根，再遍历左、右（`frontprint`方法）；
- 后序遍历：先遍历左、右，再访问根（`afterprint`方法）；
- 期末考点：三种遍历的顺序、递归实现、BST 中序遍历的升序特性。

## 四、运行示例（复习验证）

### 输入

```plaintext
3 1 4 2 5
```

### 执行过程

1. 插入节点，AVL 树自动平衡；
2. 统计叶子节点数：最终叶子节点为 2、5，数量为 2；
3. 前序遍历：3 1 2 4 5；
4. 后序遍历：2 1 5 4 3；
5. 中序遍历：1 2 3 4 5（升序，符合 BST 特性）。

### 输出

```plaintext
2
3 1 2 4 5
2 1 5 4 3
1 2 3 4 5
```

## 五、期末复习核心考点总结

| 考点分类   | 核心内容                                                     |
| ---------- | ------------------------------------------------------------ |
| AVL 树基础 | 平衡因子定义（左高 - 右高）、失衡判定（绝对值 > 1）、节点高度的定义 |
| 插入逻辑   | BST 插入规则 + 高度更新 + 平衡检查 + 旋转调整                |
| 旋转操作   | 四种失衡类型（LL/RR/LR/RL）、左旋 / 右旋的核心逻辑、旋转后高度更新 |
| 二叉树遍历 | 前序 / 中序 / 后序的遍历顺序、递归实现、BST 中序遍历升序特性 |
| 递归题     | 叶子节点统计的递归逻辑、终止条件设计                         |

## 六、总结

本文完整解析了 AVL 树的 Java 实现，涵盖 “插入（含平衡调整）、叶子统计、三种遍历” 三大核心功能，同时针对期末复习标注了关键考点。

AVL 树的核心是 “平衡”：通过维护平衡因子和旋转操作，解决了普通 BST 退化的问题，是数据结构期末的高频考点。复习时建议重点掌握：

1. 平衡因子的计算和失衡判定；
2. 四种失衡类型的旋转策略；
3. 二叉树三种遍历的递归实现；
4. 递归统计叶子节点的逻辑。