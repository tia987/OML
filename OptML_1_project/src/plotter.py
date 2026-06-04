import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms

cycler = plt.cycler(color=["darkmagenta", "palevioletred", "mediumpurple", "royalblue", "darkslateblue"])
plt.rc('axes', prop_cycle=cycler * plt.cycler(lw=[2], marker="o"))

def plot(n_generations, results_list, show=False):
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(8, 5))
    for results in results_list:
        plt.plot(range(1, n_generations + 1), results)
    plt.title('Optimizer Convergence')
    plt.xlabel('Generation / Iteration')
    plt.ylabel('Best Feasible Absorbed Energy')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join('results', 'comparison.png'))
    if show: plt.show()

def plot_plate(best_result, name='', r1=15, r2=10, slot_L=40, slot_R=10, plate_L=100):
    x1 = 30 - best_result['x1']
    y1 = 30 + best_result['y1']
    x2 = 30 + best_result['x2']
    y2 = 30 + best_result['y2']
    x3 = 30 - best_result['x3']
    y3 = 30 - best_result['y3']
    angle = best_result['angle']

    os.makedirs('results', exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Base plate
    ax.add_patch(patches.Rectangle((0, 0), plate_L, plate_L, facecolor='#f5d996', edgecolor='black', lw=2))
    
    # Circular cutouts
    ax.add_patch(patches.Circle((x1, y1), r1, facecolor='white', edgecolor='black', lw=2))
    ax.add_patch(patches.Circle((x2, y2), r2, facecolor='white', edgecolor='black', lw=2))
    
    # Centers
    ax.plot([x1, x2, x3], [y1, y2, y3], 'k.', markersize=8)
    ax.text(x1, y1, ' (x1, y1)', va='bottom')
    ax.text(x2, y2, ' (x2, y2)', va='bottom')
    ax.text(x3, y3, ' (x3, y3)', va='bottom')
    
    # Elongated cutout (Capsule)
    # Using a path for the capsule outline
    t = transforms.Affine2D().rotate_deg_around(x3, y3, angle) + ax.transData
    
    # Create the capsule path
    # Centers of the two circles
    c1_x, c1_y = x3 - slot_L/2, y3
    c2_x, c2_y = x3 + slot_L/2, y3
    
    # We can just draw the circles and the rectangle, and use a white face and black edge. 
    # To avoid internal lines, we can use a PathPatch
    from matplotlib.path import Path
    
    # vertices
    v = [
        (c1_x, y3 + slot_R), # top left
        (c2_x, y3 + slot_R), # top right
        (c2_x, y3 - slot_R), # bottom right
        (c1_x, y3 - slot_R), # bottom left
        (c1_x, y3 + slot_R)  # close
    ]
    
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    rect_path = Path(v, codes)
    
    # Actually, drawing it as overlapping shapes with no edge, then outlining it is annoying in matplotlib.
    # Let's draw the pieces with edges, and cover the internal edges with a slightly smaller white rectangle without edges.
    
    # Main shapes with edges
    rect = patches.Rectangle((c1_x, y3 - slot_R), slot_L, slot_R*2, facecolor='white', edgecolor='black', lw=2)
    rect.set_transform(t)
    ax.add_patch(rect)
    
    circ1 = patches.Circle((c1_x, y3), slot_R, facecolor='white', edgecolor='black', lw=2)
    circ1.set_transform(t)
    ax.add_patch(circ1)
    
    circ2 = patches.Circle((c2_x, y3), slot_R, facecolor='white', edgecolor='black', lw=2)
    circ2.set_transform(t)
    ax.add_patch(circ2)
    
    # Cover the middle lines with an edgeless white rectangle slightly larger horizontally
    cover = patches.Rectangle((c1_x, y3 - slot_R + 0.1), slot_L, slot_R*2 - 0.2, facecolor='white', edgecolor='none')
    cover.set_transform(t)
    ax.add_patch(cover)

    ax.set_xlim(-10, plate_L + 10)
    ax.set_ylim(-10, plate_L + 10)
    ax.set_aspect('equal')
    ax.set_title(f"Geometry Layout\nAngle: {angle}°")
    ax.axis('off')
    
    save_name = "slab_layout" + name + ".png"
    plt.savefig(os.path.join('results', save_name))
    plt.close()