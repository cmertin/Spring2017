from __future__ import print_function, division

c = 0
c_old = 0
r_x_cc = -200
r_x_cg = -400
r_x_ccp = 0.4
r_x_cgp = 0.6
r_y_cj1 = 400
r_y_cj2 = 100
r_y_cj1p = 0.4
r_y_cj2p = 0.6

g = 0
g_old = 0
r_x_gg = 100
r_x_ggp = 0.8
r_x_gj = 200
r_x_gjp = 0.2
r_y_gj = 1000
r_y_gjp = 1.0

j = 0
j_old = 0

gamma = 0.5

for i in range(10):
    c1 = r_x_ccp * (r_x_cc + gamma**i * c_old) + r_x_cgp * (r_x_cg + gamma**i * c_old)
    c2 = r_y_cj1p * (r_y_cj1 + gamma**i * c_old) + r_y_cj2p * (r_y_cj2 + gamma**i * c_old)
    c = max(c1, c2)

    g1 = r_x_ggp * (r_x_gg + gamma**i * g_old) + r_x_gjp * (r_x_gj + gamma**i * g_old)
    g2 = r_y_gjp * (r_y_gj + gamma**i * g_old)
    g = max(g1, g2)
    c_str = "%.1f" % c
    g_str = "%.1f" % g
    print("itr = " + str(i + 1), c_str, g_str)

    c_old = c
    g_old = g
