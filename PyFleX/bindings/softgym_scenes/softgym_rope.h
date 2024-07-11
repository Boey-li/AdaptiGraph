
class SoftgymRope : public Scene
{
public:

	SoftgymRope(const char* name) : Scene(name) {}

    // init_x, init_y, init_z, stiffness, segments, length, invmass
	void Initialize(py::array_t<float> scene_params, 
                    py::array_t<float> vertices,
                    py::array_t<int> stretch_edges,
                    py::array_t<int> bend_edges,
                    py::array_t<int> shear_edges,
                    py::array_t<int> faces,
                    int thread_idx = 0)
	{

        auto ptr = (float *) scene_params.request().ptr;
        float init_x = ptr[0];
        float init_y = ptr[1];
        float init_z = ptr[2];
        float stretchstiffness = ptr[3];
        float bendingstiffness = ptr[4];
        float radius = ptr[5]; // used to determine the num of segments
        float segment = ptr[6];
        float mass = ptr[7];
        float scale = ptr[8];
		float draw_mesh = ptr[9];

		int group = 0;

        Rope r;

        Vec3 d0 = Vec3(0, 1, 0);
        Vec3 start = Vec3(init_x, init_y, init_z);

		// void CreateRope(Rope& rope, Vec3 start, Vec3 dir, float stiffness, 
		//      int segments, float length, int phase, float spiralAngle=0.0f, 
		//		float invmass=1.0f, float give=0.075f, float bendingStiffness=0.8)
		// phase = NvFlexMakePhase(group++, eNvFlexPhaseSelfCollide | eNvFlexPhaseSelfCollideFilter);
        CreateRope(r, start, d0, stretchstiffness, int(segment), segment * radius * 0.5, 
            NvFlexMakePhase(group++, 0), 0.0f, 1 / mass, 0.075f, bendingstiffness);

        g_ropes.push_back(r);
		

	   	g_params.radius = radius;
		// g_params.numIterations = 4;
		// g_params.dynamicFriction = 1.0f;
		// // g_params.staticFriction = 0.8f;
		// g_params.collisionDistance = 0.001f;
		
		// g_maxDiffuseParticles = 64*1024;
		// g_diffuseScale = 0.25f;		
		// g_diffuseShadow = false;
		// g_diffuseColor = 2.5f;
		// g_diffuseMotionScale = 1.5f;
		// g_params.diffuseThreshold *= 0.01f;
		// g_params.diffuseBallistic = 35;

		g_params.fluidRestDistance = radius;
		g_params.numIterations = 4;
		g_params.viscosity = 1.0f;
		g_params.dynamicFriction = 0.05f;
		g_params.staticFriction = 0.0f;
		g_params.particleCollisionMargin = 0.0f;
		g_params.collisionDistance = g_params.fluidRestDistance*0.5f;
		g_params.vorticityConfinement = 120.0f;
		g_params.cohesion = 0.0025f;
		g_params.drag = 0.06f;
		g_params.lift = 0.f;
		g_params.solidPressure = 0.0f;
		g_params.smoothing = 1.0f;
		g_params.relaxationFactor = 1.0f;

		g_windStrength = 0.0f;
		g_windFrequency = 0.0f;

		g_numSubsteps = 2;

		// draw options		
		// g_drawEllipsoids = false;
		// g_drawPoints = false;
		// g_drawDiffuse = false;
		// g_drawSprings = 0;

		if (!draw_mesh) {
            g_drawPoints = true;
            g_drawMesh = false;
            g_drawRopes = false;
            g_drawSprings = true;
        }
        else {
		    g_drawPoints = false;
		    g_drawMesh = true;
		    g_drawRopes = true;
		    g_drawSprings = false;
		}

		g_ropeScale = scale;
		g_warmup = false;
	}
};



