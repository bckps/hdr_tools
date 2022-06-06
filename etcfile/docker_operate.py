import docker

camera_samples = ["-camera-spp", 4]
bdpt_samples = ["-bidirectional-path-tracing", 512]
bounce_streak = ["-multibounce-streak", 3]
max_bounces = ["-max-nb-bounces", 8]
height = ["-film-size-y", 257]
width = ["-film-size-x", 256]
time_res = ["-film-size-t", 2220]
toffset = ["-film-offset", 0]
exp = ["-film-exposure", 0.008993773740]

camera_pos = ["-camera-position", '0.7988916 0.2083092 1.765791']
camera_at = ["-camera-focus", '0.39171684 0.5006809  0.90050059']
camera_lookup = ["-camera-up", '0.92337742 1.16461396 2.03033683']
camera_fov = ["-camera-fov", 40]

point_light = ["-point-light-source", '0.7988916 0.2083092 1.865791 0.1']

objpath = '/home/saijo/labwork/simulator_origun/model/export_bathroom_small.obj'
objfile = ["-name-mesh", objpath, "-lambertian 0.9 0.9 0.9"]
# Don't use underbar in name
name = 'python-docker-bathroom-small'
log = ["-log-name", name+'.txt']
filmname = ["-film-name", name]

configlist = [
    *["BunnyKiller"],
    *log,
    *filmname,
    *camera_samples,
    *camera_fov,
    *width,
    *height,
    *time_res,
    *exp,
    *toffset,
    *camera_pos,
    *camera_at,
    *camera_lookup,
    *point_light,
    *['-scattering-level all -lambertian-rho 1 -no-background'],
    *objfile,
    *bdpt_samples,
    *bounce_streak,
    *max_bounces,
    *['-transient-state'],
]

client = docker.from_env()
print(client.containers.run("bunnykiller", configlist))