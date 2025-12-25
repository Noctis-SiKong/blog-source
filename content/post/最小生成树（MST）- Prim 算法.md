---
title: '最小生成树（MST）'
date: '2025-12-25T19:30:10+08:00'
draft: false
series: '数据结构与算法优化'
---

## 最小生成树（MST）- Prim 算法

### 核心场景（详细版）

最小生成树（MST）适用于**无向连通图**，核心目标是：找到一组边，满足「连接所有顶点」且「边权和最小」，同时无环。

- 典型应用：村村通公路（最低成本连接所有村庄）、网络布线（最低成本连接所有节点）；

- 额外需求：若图不连通，无法生成 MST，需输出 - 1（表示需新增边）。

  <!--More-->

### 分类一：算法思想（详细拆解，Prim 算法 —— 适合稠密图）

Prim 算法基于「贪心策略」，核心是 “逐步扩展 MST，每次选连接 MST 和非 MST 的最小权值边”，步骤拆解如下：

1. **核心概念定义**：
   - MST 集合：已加入最小生成树的顶点集合；
   - 距离数组 dist：dist [v] 表示「顶点 v 到 MST 集合的最小边权」（而非到起点的距离，区别于 Dijkstra）。
2. **算法步骤**：
   - 步骤 1：初始化：
     - 任选一个起点（如顶点 1），dist [start] = 0；
     - 其余 dist [v] = INF（初始到 MST 无连接）；
     - visited 数组全为 false（未加入 MST）。
   - 步骤 2：循环 n 次（n = 顶点数）：
     - 子步骤 1：找「未访问且 dist 最小」的顶点 u（该顶点是连接 MST 的最优选择）；
     - 子步骤 2：若 u=-1（无可达顶点），说明图不连通，返回 - 1；
     - 子步骤 3：将 u 加入 MST，累加 dist [u] 到总权值，标记 visited [u] = true；
     - 子步骤 4：更新 u 的邻接顶点 v 的 dist：若 u→v 的边权 w <dist [v]，则 dist [v] = w（更新 v 到 MST 的最小边权）。
   - 步骤 3：结果判断：若加入 MST 的顶点数≠n，说明图不连通，返回 - 1；否则返回总权值。

### 通用精简代码

```java
import java.util.*;

public class Prim {
    public static final int INF = Integer.MAX_VALUE;

    // 通用Prim：返回MST总权值，不连通返回-1
    // 参数：n=顶点数（1~n），adj=邻接表（adj[u]存储{v, w}）
    public static int prim(int n, List<List<int[]>> adj) {
        boolean[] visited = new boolean[n + 1]; // 1基
        int[] dist = new int[n + 1];
        Arrays.fill(dist, INF);
        dist[1] = 0; // 起点选1（任选，不影响结果）

        int total = 0; // MST总权值
        int count = 0; // 加入MST的顶点数

        for (int i = 0; i < n; i++) {
            // 步骤1：找未访问的dist最小顶点u
            int u = -1;
            int minDis = INF;
            for (int j = 1; j <= n; j++) {
                if (!visited[j] && dist[j] < minDis) {
                    minDis = dist[j];
                    u = j;
                }
            }
            if (u == -1) return -1; // 不连通

            // 步骤2：加入MST，累加权值
            visited[u] = true;
            total += minDis;
            count++;

            // 步骤3：更新邻接顶点的dist（到MST的最小边权）
            for (int[] edge : adj.get(u)) {
                int v = edge[0];
                int w = edge[1];
                if (!visited[v] && w < dist[v]) {
                    dist[v] = w;
                }
            }
        }
        return count == n ? total : -1;
    }

    // 复用入口（适配"公路村村通""畅通工程2"）
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        while (sc.hasNext()) {
            int n = sc.nextInt(); // 顶点数
            if (n == 0) break; // 终止条件
            int m = n == 1 ? 0 : sc.nextInt(); // 边数（n=1时无需边）

            // 初始化邻接表（1基）
            List<List<int[]>> adj = new ArrayList<>();
            for (int i = 0; i <= n; i++) adj.add(new ArrayList<>());
            // 处理边输入（适配不同题目格式）
            int edgeCount = n == 1 ? 0 : (n*(n-1)/2); // 畅通工程2的边数
            edgeCount = m == 0 ? edgeCount : m; // 公路村村通用m
            for (int i = 0; i < edgeCount; i++) {
                int u = sc.nextInt();
                int v = sc.nextInt();
                int w = sc.nextInt();
                // 无向图：双向添加边
                adj.get(u).add(new int[]{v, w});
                adj.get(v).add(new int[]{u, w});
            }

            // 执行Prim
            int res = prim(n, adj);
            System.out.println(res == -1 ? -1 : res);
        }
        sc.close();
    }
}
```

### 核心考点（详细解释）

1. **Prim vs Dijkstra 的区别**：
   - Prim 的 dist [v]：顶点 v 到 MST 的最小边权；
   - Dijkstra 的 dist [v]：顶点 v 到起点的最短路径长度；
   - 两者贪心策略相同（选最小 dist 的顶点），但 dist 的含义完全不同；
2. **起点选择**：任选一个顶点作为起点（如 1），最终 MST 的总权值相同；
3. **稠密图适配性**：Prim 算法（基础版 O (n²)）适合稠密图（边数多），Kruskal 算法适合稀疏图；
4. **连通性判断**：count（加入 MST 的顶点数）==n → 连通，否则不连通。

#### 易错点（详细说明错误原因 + 正确做法）

| 易错点                | 错误原因                      | 正确做法                                   |
| --------------------- | ----------------------------- | ------------------------------------------ |
| 混淆 Prim 和 Dijkstra | dist 数组含义理解错误         | 牢记 Prim 的 dist 是 “到 MST 的最小边权”   |
| 无向图未双向加边      | 邻接表漏边，导致 MST 无法生成 | 必须同时添加 u→v 和 v→u 的边               |
| 顶点编号未适配 1 基   | 数组越界（题目顶点从 1 开始） | 数组开到 n+1，遍历从 1 开始                |
| 起点 dist 未设 0      | 第一个顶点无法被选中          | 强制设置 dist [start] = 0（如 dist [1]=0） |

### 总结（辅助记忆）

Prim 算法核心口诀：「选最小边权点→加入 MST→更新邻接边权→统计顶点数判连通」；

核心区别：dist 数组是 “到 MST 的最小边权”（而非到起点的距离）；

核心适用：无向稠密图、求最小权值连通边集。



## 分类二：最小生成树（MST）- Kruskal 算法

### 核心场景（详细版）

最小生成树（MST）适用于**无向连通图**，核心目标与 Prim 算法一致：找到一组边，满足「连接所有顶点」且「边权和最小」，同时无环。

- 典型应用：与 Prim 算法相同（如村村通公路、网络布线），但更适用于**稀疏图**（边数少的场景）；
- 额外需求：若图不连通，无法生成 MST，需输出 -1（表示需新增边）。

### 算法思想（详细拆解，Kruskal 算法 —— 适合稀疏图）

Kruskal 算法同样基于「贪心策略」，核心是 “从边的角度选最小权值边，避免形成环”，步骤拆解如下：

1. **核心概念定义**：
   - 边集排序：所有边按权值从小到大排序；
   - 并查集（Union-Find）：用于高效判断「添加一条边是否会形成环」（检测两顶点是否已在同一连通分量）。
2. **算法步骤**：
   - 步骤 1：初始化：
     - 将所有边按权值升序排序；
     - 初始化并查集（每个顶点独立成树）；
     - 定义变量 total 记录 MST 总权值，count 记录加入 MST 的边数（最终需为 n-1，n 为顶点数）。
   - 步骤 2：遍历排序后的边：
     - 子步骤 1：取当前权值最小的边（u, v, w）；
     - 子步骤 2：用并查集判断 u 和 v 是否在同一连通分量：
       - 若不在：将边加入 MST，total += w，count += 1，合并 u 和 v 所在的集合；
       - 若在：跳过（避免形成环）。
   - 步骤 3：结果判断：若 count == n-1（所有顶点连通），返回 total；否则返回 -1（图不连通）。

### 通用精简代码

```java
import java.util.*;

public class Kruskal {
    // 并查集实现（用于检测环和合并集合）
    static class UnionFind {
        int[] parent;
        int[] rank; // 按秩合并优化

        public UnionFind(int n) {
            parent = new int[n + 1]; // 1基
            rank = new int[n + 1];
            for (int i = 1; i <= n; i++) {
                parent[i] = i; // 自身为父节点
                rank[i] = 1; // 初始秩为1
            }
        }

        // 查找根节点（路径压缩优化）
        public int find(int x) {
            if (parent[x] != x) {
                parent[x] = find(parent[x]); // 路径压缩
            }
            return parent[x];
        }

        // 合并两个集合（按秩合并）
        public boolean union(int x, int y) {
            int rootX = find(x);
            int rootY = find(y);
            if (rootX == rootY) return false; // 已在同一集合（会形成环）

            // 秩小的树合并到秩大的树
            if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
            } else if (rank[rootX] > rank[rootY]) {
                parent[rootY] = rootX;
            } else {
                parent[rootY] = rootX;
                rank[rootX]++;
            }
            return true;
        }
    }

    // 通用Kruskal：返回MST总权值，不连通返回-1
    // 参数：n=顶点数（1~n），edges=边列表（每个元素为{u, v, w}）
    public static int kruskal(int n, List<int[]> edges) {
        // 步骤1：边按权值升序排序
        Collections.sort(edges, (a, b) -> a[2] - b[2]);

        UnionFind uf = new UnionFind(n);
        int total = 0; // MST总权值
        int count = 0; // 加入MST的边数（需达到n-1）

        // 步骤2：遍历所有边
        for (int[] edge : edges) {
            int u = edge[0];
            int v = edge[1];
            int w = edge[2];

            // 若u和v不在同一集合，加入边并合并
            if (uf.union(u, v)) {
                total += w;
                count++;
                // 已收集足够边（n-1条），提前退出
                if (count == n - 1) break;
            }
        }

        // 步骤3：判断是否连通
        return count == n - 1 ? total : -1;
    }

    // 复用入口（适配稀疏图场景）
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        while (sc.hasNext()) {
            int n = sc.nextInt(); // 顶点数
            if (n == 0) break;
            int m = sc.nextInt(); // 边数（稀疏图边数少，直接输入）

            List<int[]> edges = new ArrayList<>();
            for (int i = 0; i < m; i++) {
                int u = sc.nextInt();
                int v = sc.nextInt();
                int w = sc.nextInt();
                edges.add(new int[]{u, v, w});
            }

            // 执行Kruskal
            int res = kruskal(n, edges);
            System.out.println(res == -1 ? -1 : res);
        }
        sc.close();
    }
}
```

### 核心考点（详细解释）

1. **Kruskal vs Prim 的区别**：
   - 适用场景：Kruskal 适合**稀疏图**（边数少，排序成本低），Prim 适合**稠密图**（顶点少，邻接表遍历高效）；
   - 核心工具：Kruskal 依赖「并查集」检测环，Prim 依赖「距离数组」跟踪最小边权；
   - 时间复杂度：Kruskal 为 O (m log m)（m 为边数，主要耗时在排序），Prim（基础版）为 O (n²)（n 为顶点数）。
2. **并查集的作用**：
   - 高效判断两顶点是否连通（避免环）；
   - 路径压缩和按秩合并优化后，单次操作接近 O (1)。
3. **边的处理**：
   - 必须先按权值排序（贪心选择基础）；
   - 无需处理自环（排序后也会被并查集检测为同集合，自动跳过）。

#### 易错点（详细说明错误原因 + 正确做法）

| 易错点             | 错误原因                              | 正确做法                                              |
| ------------------ | ------------------------------------- | ----------------------------------------------------- |
| 边未排序或降序排序 | 无法保证选到最小权值边                | 严格按边权升序排序（`Collections.sort` 用升序比较器） |
| 并查集未优化       | 大规模数据时超时（查找 / 合并效率低） | 必须实现「路径压缩」和「按秩合并」优化                |
| 忽略连通性判断条件 | 误判非连通图为连通                    | 严格检查加入的边数是否为 `n-1`（而非顶点数）          |
| 处理有向图         | Kruskal 仅适用于无向图                | 输入时确保边是无向的（无需额外处理，算法本身兼容）    |
| 边的存储格式错误   | 数组索引与顶点 / 权值对应关系混乱     | 统一用 `{u, v, w}` 格式存储边信息                     |

### 总结（辅助记忆）

Kruskal 算法核心口诀：「边排序→查环（并查集）→加边→边数判连通（n-1）」；

核心区别：依赖并查集处理边，适合边数少的稀疏图；

核心适用：无向稀疏图、边权可排序场景、需高效处理大规模顶点（顶点多但边少）。