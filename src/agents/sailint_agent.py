"""
WindAwareNavigator - Agent avancé pour le Sailing Challenge

Cet agent combine:
1. Algorithme A* adapté à la navigation à voile pour une planification optimale de la trajectoire
2. Réseaux de neurones pour l'évaluation des états complexes et les prédictions de vent
3. Analyse sophistiquée du champ de vent avec clustering et détection de courants
4. Stratégies adaptatives basées sur la physique de la voile

L'agent est capable de:
- Planifier des trajectoires optimales en tenant compte des contraintes de la voile
- Apprendre des modèles complexes du comportement du vent et du bateau
- Identifier et exploiter des structures de vent avantageuses
- S'adapter rapidement aux changements de conditions
"""

import numpy as np
import heapq
from collections import defaultdict, deque
from agents.base_agent import BaseAgent

class WindAwareNavigator(BaseAgent):
    """
    Agent avancé pour le Sailing Challenge qui combine A*, réseaux de neurones et 
    analyse sophistiquée du champ de vent.
    """
    
    def __init__(self):
        """Initialise l'agent avec les paramètres et modèles nécessaires."""
        super().__init__()
        self.np_random = np.random.default_rng()
        
        # Paramètres de l'environnement
        self.grid_size = (32, 32)
        self.goal_position = np.array([self.grid_size[0] // 2, self.grid_size[1] - 1])
        
        # Paramètres de la physique de la voile
        self.optimal_beam_reach_angle = np.pi/2  # 90 degrés, perpendiculaire au vent
        self.close_hauled_angle = np.pi/4        # 45 degrés, près du vent
        self.no_go_zone_angle = np.pi/6          # 30 degrés, zone interdite
        
        # Initialisation des modèles de réseaux de neurones simplifiés
        self.initialize_neural_networks()
        
        # État interne
        self.previous_position = None
        self.steps_without_progress = 0
        self.last_distance_to_goal = float('inf')
        self.current_tack_direction = 1  # 1 pour tribord, -1 pour bâbord
        self.path_cache = {}  # Cache pour les chemins A*
        self.current_path = []  # Chemin actuel suivi
        
        # Mémoire des vents pour analyse de tendances
        self.wind_memory = deque(maxlen=10)
        
        # Paramètres de stratégie et modes
        self.exploration_rate = 0.05
        self.use_astar = True
        self.use_nn_evaluation = True
        self.use_wind_analysis = True
        
        # Seuils et variables de contrôle
        self.path_recompute_threshold = 5  # Nombre d'étapes avant de recalculer le chemin
        self.steps_since_path_recompute = 0
        self.prediction_horizon = 10  # Nombre d'étapes pour prédire l'évolution du vent
        
    def initialize_neural_networks(self):
        """Initialise les réseaux de neurones simplifiés (weights-only)."""
        # Modèle d'évaluation d'état : évalue l'optimalité d'une position donnée
        # Architecture : entrée[position(2), vitesse(2), vent(2)] -> couches cachées -> valeur(1)

        self.state_eval_model = {
            'W1': np.array([[0.019313364376407044, 0.07934115482951021, 0.017396672756038453, -0.10299534038410728, 0.05893156682027095, -0.0681874201828102, -0.14010442967953982, -0.11421291749560421, -0.09914004637651597, -0.035441075079443024], 
                            [-0.13237239038467008, 0.0598859977426523, 0.07538206218160674, -0.12002855333819118, 0.024079871823161726, -0.12112656817088392, 0.014243232929662572, 0.08566733733954174, 0.015753045165461936, 0.06374540965957134], 
                            [0.1687005167865639, 0.07212640821822593, -0.08974594453339996, -0.12039274585781268, 0.034650782550000536, -0.16453279286629507, 0.11882600324253059, -0.06294800504889238, -0.32184112864687053, -0.03269530192724545], 
                            [0.00906422636656765, -0.0899744828355049, -0.013391636323646481, 0.23445806930625426, 0.06891364430260084, 0.29055719860236423, 0.005583885399469348, 0.16840018954968594, -0.06860843952301734, -0.028529773878093783], 
                            [-0.0446081372592477, 0.02576882182157529, -0.009285763254401006, 0.21202599732017788, -0.03952236600414781, -0.015723572218613105, -0.0352392792274916, -0.03309145256900489, -0.18336888213799496, 0.07392900016221926], 
                            [-0.024398638448289676, 0.0650679299606748, -0.009608589517322852, -0.02671521628832947, -0.027172089441610616, 0.02972093962181583, -0.026912602262430932, 0.005410941514247472, -0.15831335359166712, -0.11647512702755523]]),
            'b1': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'W2': np.array([[0.07608107647861835, 0.10116886234240699, 0.49916211105831565, -0.06316814040436076, -0.11952386586972166], 
                            [-0.08580757057443646, -0.057971444329004584, 0.5004641311681709, 0.10889816823497492, 0.03727372707505942], 
                            [-0.11036026604952068, 0.2503479257337791, 0.5004248432373212, 0.2489409388890358, 0.10038505191242397], 
                            [-0.01464062358327732, 0.030172547650851922, 0.4996414871537795, 0.1788375159898971, 0.10722810463669769], 
                            [0.08334569319617721, -0.22119232515129125, 0.500137327454649, -0.19434306077541091, 0.06907971838501914], 
                            [-0.06891081425259037, 0.03642191061595198, 0.4989691073288318, -0.030268108721859182, 0.13410350447447958], 
                            [0.09165054163671267, 0.09799787077387175, 0.5001452303717651, 0.15827424805672716, -0.09045341650585273], 
                            [-0.013710170888739017, 0.030282331414131395, 0.500451669277978, -0.13860927869751904, -0.06490195932831215], 
                            [-0.03700122974729947, -0.03462335304161716, 0.4995277759782853, 0.2012453364616848, -0.043705223316994925], 
                            [-0.06752538407027159, -0.14600534929793046, 0.5007279906359507, 0.1861692580707159, 0.15550105582252857]]),
            'b2': np.array([0.0008001937345360987, -0.002990070107138212, 0.0003943000560917304, -0.0035090001027599533, -0.0002666077288247823]),
            'W3': np.array([[0.04594707993416697], 
                            [-0.2078667466011548], 
                            [0.02588197088484641], 
                            [-0.2384677446394869], 
                            [-0.009446704810004437]]),
            'b3': np.array([0.14433060646853257]),
        }
    
    # Modèle de prédiction de vent : prédit l'évolution du vent à partir des observations passées
        self.wind_prediction_model = {
            'W1': np.array([[-0.15263704416981894, 0.009318413415495093, 0.01788566281886031, 0.04338565370468317, -0.039545128620585235, 0.012844375555425534, -0.10895468239396489, 0.13596216986746398, -0.10455694553988788, -0.10794579868285113, -0.14902587706816325, -0.2193704878493589, 0.14458408451820634, 0.12912849345236885, 0.2582094359017618], 
                            [-0.010937521260260798, 0.01948847963837895, 0.012236740041533956, -0.0098583755276049, -0.06548309050498918, 0.07366818354080922, -0.023303556643905484, 0.013108160788129548, -0.016831128096055567, 0.16745026512167135, 0.10813845645839522, 0.1612794203519638, -0.03629923246455525, 0.021702925310289098, -0.045629585704958994], 
                            [0.08027099941365216, -0.0861376888059602, 0.10674628197288423, 0.13125870284231722, 0.11005231780971021, 0.07082631801321539, 0.019423106652471656, 0.18065445264979363, 0.026826214640731302, 0.18587203620775308, -0.13459599699697775, 0.20156799226989075, -0.059877358963857374, -0.08749984038893582, -0.07191823822924663], 
                            [-0.18514455849336142, -0.03298684748677184, 0.09047188209201208, 0.06497041900980548, 0.02066487012735469, -0.16681115971564664, 0.03603960530014032, 0.006230576993330044, 0.07528945959200761, -0.17110184296794972, 0.2156627322430682, 0.11671082472166064, -0.07923067416426421, 0.040686416819966154, -0.0882662587622846], 
                            [0.05026250028427412, 0.09792234586146797, -0.08684463119790015, -0.13306772871559866, -0.0856957423609675, 0.011534431898405223, 0.00864803989780255, -0.10887107146928146, 0.17063189738826082, 0.047263137704048885, -0.030654454268150395, -0.23094435485419834, 0.053538856609890786, 0.01581333299942396, 0.07778169699657772], 
                            [0.00620075706876153, 0.1287895371918549, 0.08357749890571042, -0.051522946844333874, -0.10937336784080559, 0.04171818508772923, 0.0019202855761981516, 0.1714695387088082, -0.1156369098138971, -0.03256534908656869, 0.08518991858298743, 0.023370541665146086, -0.03539896762840767, -0.2224084461319918, -0.0006500880817091808], 
                            [0.2568922201653999, -0.10981198832284494, -0.0585260098020611, -0.052832303713701026, -0.033929102951898236, -0.13671856118511103, -0.002456008249260377, 0.03283829368844886, -0.10082350026607872, -0.021813792033086088, -0.031027155016415398, 0.13407763999265332, -0.050021767864129096, 0.16731354236182355, 0.08458630865191391], 
                            [-0.16019714261321216, -0.04630709159741453, -0.034663530350262844, 0.0445341479778971, 0.030165453298138186, 0.04678956746369994, 0.02235682486759332, -0.023533552449432445, 0.022230572226972268, -0.12659083763220286, -0.003653723460494027, -0.09663237648205758, -0.1971226946400165, -0.11266691656908717, 0.0756004139381154], 
                            [-0.025981492327851352, 0.02672060059815727, -0.06503930353247526, -0.08787743935397438, -0.0520555455933692, 0.013724166031046068, -0.06095623903089567, 0.06082815611119885, 0.07133068428526922, 0.02656299961568133, 0.01211046176940125, -0.10909017575390717, -0.1903449676205773, 0.03774255444450102, 0.16532189005050008], 
                            [-0.07124033878071201, 0.025900156856566714, 0.03703070586203098, -0.153868693131401, -0.030246786444185644, -0.24400846312325483, 0.017343623171547442, -0.08357325115404574, -0.0914704206133485, 0.03955272324991493, -0.10515288616834338, -0.20144760916722015, -0.03873687735976469, 0.20895181304717309, 0.10043309458562025], 
                            [-0.021023369161981584, -0.018885551997644412, 0.1740178868643283, -0.1335038007841269, -0.09401809290787654, -0.05950537258194902, 0.07527109606610849, 0.0009177507938848133, 0.04343486396260307, 0.038877270243503806, 0.08189892577909899, -0.15304924472707282, 0.13803813453478667, 0.04483066446464684, 0.03588157637929124], 
                            [-0.0745761776922403, 0.0876607431021326, -0.08234135574935718, 0.012593576401686622, 0.05602908085220113, 0.1300382434372369, -0.1878991758244089, 0.11627026883473371, 0.05874130515925378, -0.007130728849099184, -0.026707982064129415, 0.17379436815373275, 0.2592186992665671, 0.03289215188388898, 0.16535572513885258], 
                            [-0.0033663774153866533, 0.13751822185238743, 0.039149839016744896, 0.011297538069006578, 0.06212231617216962, 0.06129921180311441, -0.08115402451058715, -0.12844942454194955, -0.3535134915544022, 0.09438708298702576, -0.026263883881117545, 0.049099344117154727, 0.004177955904516242, 0.0628923879171176, -0.08488966070283849], 
                            [-0.052308172711140716, -0.062451011143833494, -0.0741511910139186, -0.1323806292931067, -0.07581781242343545, 0.1213774631404999, 0.07134126553170672, 0.0770623154464379, 0.0713083930758758, 0.027891515047747163, -0.11531649964983762, -0.015214636455490711, -0.022346495586645643, -0.1395072021332262, -0.007623056476835629], 
                            [-0.17459279840640185, 0.09460919725718213, -0.014386907974766739, -0.05050236115390924, 0.05236879235569032, 0.11645161687477945, -0.02790527309319768, 0.042879972970464926, -0.055101237702648034, 0.029024697323369587, 0.018449394799917324, -0.07360066856888114, -0.018424019276114516, 0.02031133319236647, 0.03040051192558408], 
                            [0.15285114118349513, -0.13276338385001127, -0.19994548978211393, -0.04511125555140754, -0.04604699050383614, 0.05201331885413668, -0.08267184236197889, -0.1472863152089607, 0.07130968127080826, -0.04796656260846005, -0.11674268175244742, 0.15146940256148134, 0.15395520632703008, 0.13688140200542412, -0.045260210347369795], 
                            [-0.08999678563068661, -0.14626517209982398, -0.14291092739771852, 0.00018416041168860744, -0.09925037498565953, -0.10830244801776041, 0.12347725769804599, -0.2032514824103879, -0.10730261125867627, -0.06642157100792875, 0.08986592351851998, -0.03996905930523789, 0.02156273670681741, -0.050219142878923276, -0.2291694968191243], 
                            [-0.13294798348692857, 0.07375357941156133, 0.18746037135208027, -0.04253129795256756, 0.03745569385243895, 0.08278179423342258, -0.010954734558052126, -0.0008825415079235311, -0.14301968188769365, -0.041023740681468975, 0.02282866912061501, 0.1970942123069085, 0.05783279579123242, 0.013952309448433613, 0.19404423706775617], 
                            [-0.12150463772394929, -0.04038841654511722, -0.11100463573435503, 0.0679588316575573, 0.07552611837380252, -0.08454345488909143, 0.03810686156013305, 0.1478532132085191, -0.0630239638537125, 0.01648529739322663, -0.04812505893047183, 0.18274260597905512, 0.12045132893004887, 0.02108479517997086, 0.0234223972410669], 
                            [0.18593926196963878, -0.08654074297911356, 0.01826571878445347, 0.022235389374056957, -0.19573114388094065, 0.023548634996002186, -0.09328523818933322, 0.015835782749868462, -0.047133419776735797, -0.0693955358414133, -0.07416904926086501, -0.03674555416045543, -0.01777780530465124, -0.018024648918754195, 0.07458698254721129]]),
            'b1': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'W2': np.array([[0.09488163823369669, 0.1350431234200042, -0.006498574130106842, -0.04301536562886879, -0.13794694581973566, 0.06939569634468559, 0.015912908157313808, 0.09544373901428876, 0.06776052412064339, -0.06486977656189634], 
                            [-0.21412606250215227, -0.07164332553591277, -0.06559203302341889, 0.09787053712538717, 0.07212280426184124, -0.1227026709381398, -0.01448571744041631, 0.13579182098403048, -0.11384269720374314, -0.11602305855311781], 
                            [0.058368242698982, -0.03253028254661827, -0.06692746235329984, -0.014715448484939556, -0.1413542111871304, 0.060734157587625716, -0.017398288964794315, 0.10789157677069415, -0.11113754746419076, 0.06018300441187666], 
                            [0.23336841239412218, -0.17687816416094265, -0.016795633897954642, -0.01998301864808269, 0.0031659919252541614, 0.08513533939257092, 0.11778121953160714, -0.14893597058060046, -0.082965169028802, 0.05772648686299971], 
                            [0.1216163666934978, 0.04815685828651039, 0.0657001217810824, -0.20519692904883446, -0.11601725228885931, 0.08194448253307862, -0.14390281114456163, -0.11716879608127358, 0.1950001926650249, -0.18545105673821866], 
                            [0.16654498507360566, 0.029316188390589488, 0.08158297843069862, -0.21371350663752742, -0.22703930311618684, 0.20722595775929267, 0.08507896099581029, -0.030671669435360385, 0.022553232710240498, 0.06232846197361633], 
                            [-0.026923206059318746, -0.028480733467960446, 0.09394879586025123, 0.12421902533830648, 0.036936412989803886, -0.006001346287808881, -0.14588488346674383, -0.051613599260382406, -0.12101578255028358, 0.11489998448045811], 
                            [-0.0375636013465569, 0.015979130273596157, -0.03101885375628278, -0.007957008757648652, -0.05023379327325048, -0.16874127762780886, -0.13688102880735278, 0.126159507826171, 0.2144273392019338, 0.011762048415610805], 
                            [0.0818942010140187, -0.16619072250331368, 0.12727899261438622, -0.0768511098476028, 0.3098139697370428, 0.030616619006504715, -0.2060782814756862, 0.08752422321964665, -0.05129537622992697, 0.055642483297362016], 
                            [-0.0032419540492792244, -0.041424938565006786, 0.03548062673084101, -0.05518569400180684, 0.08501199619975512, 0.06133331433161999, -0.07888488079341222, -0.040076003569690447, 0.016328905363887883, -0.11624031540771192], 
                            [0.024422452659546543, -0.09571028500840213, -0.047119930004666216, -0.022725226142427852, 0.030950694347471143, 0.10965672316042, 0.015631518029143702, -0.02936803886923798, 0.22829758754397675, 0.03979620945333599], 
                            [-0.08111018843935501, 0.03200561510581483, -0.0352903726372589, -0.008377585520605298, 0.0019910949297185154, 0.036680038090906225, -0.21843960965350995, 0.08375046383615165, 0.12743079049572478, 0.07179467382111249], 
                            [0.07349280509617467, -0.03868324757139568, -0.02542653912119079, 0.056167481425987535, -0.00400476832915929, 0.17166665495725378, -0.08506554906706684, 0.04834016601424887, -0.0976537036211525, -0.10574830603543016], 
                            [-0.048885269708226875, 0.0508929105919228, 0.05000474287109274, 0.03738243290258885, -0.08504523244959604, 0.021412192629632357, 0.07462634608393608, 0.047813665134880196, -0.20236592541579634, 0.056415569695042536], 
                            [0.0952440550023677, 0.05678911760020611, 0.10848012636586837, 0.044826984197053506, -0.12576590078315522, 0.01678165880467072, -0.11477529028487408, -0.0926301807660702, 0.05126715214407518, -0.15987660688285626]]),
            'b2': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'W3': np.array([[0.03808084709560377, 0.01546473431727345], 
                            [0.017696078758047844, 0.051592702369045874], 
                            [0.03793890976564407, 0.11417316684529198], 
                            [-0.12497712316642028, 0.018016633418852377], 
                            [0.030694148414271934, -0.1386784212318724], 
                            [-0.07191250744722598, -0.08981825052550915], 
                            [-0.0018800237401687223, 0.10681745133824108], 
                            [0.01677524012640202, 0.07147231747341624], 
                            [0.125648957395279, -0.04633534904835249], 
                            [-0.12423896051228256, 0.04667346016709796]]),
            'b3': np.array([0.0, 0.0]),
        }        
        # Dans un cas réel, ces poids seraient entraînés via backpropagation
        # Ici, nous initialisons avec des valeurs qui produisent des comportements raisonnables
        # en fonction de la physique de la voile et des conditions de vent
        #self._adjust_weights_for_sailing_physics()

    def _adjust_weights_for_sailing_physics(self):
        """Ajuste les poids des réseaux pour intégrer les connaissances de la physique de la voile."""
        # Exemples d'ajustements pour le modèle d'évaluation d'état
        # Ces ajustements sont basés sur des heuristiques connues de la navigation à voile
        
        # Favoriser les positions perpendiculaires au vent (beam reach)
        self.state_eval_model['W1'][2:4, :] *= 1.5  # Augmenter l'influence du vent
        
        # Pénaliser l'évaluation des états avec vitesse faible
        self.state_eval_model['W2'][:, 2] = 0.5  # Poids plus importants pour la vitesse
        
        # Ajustements pour le modèle de prédiction de vent
        # Intégrer un biais pour les tendances typiques du vent (rotations lentes)
        self.wind_prediction_model['W2'][5:10, 3:7] *= 1.2  # Renforcer certains patterns de changement de vent
        
    def nn_forward(self, x, model):
        """Propagation avant simplifiée pour un réseau de neurones à 3 couches."""
        h1 = np.tanh(np.dot(x, model['W1']) + model['b1'])
        h2 = np.tanh(np.dot(h1, model['W2']) + model['b2'])
        out = np.dot(h2, model['W3']) + model['b3']
        return out
    
    def evaluate_state(self, position, velocity, wind):
        """Évalue la qualité d'un état en utilisant le réseau de neurones."""
        # Normaliser les entrées
        pos_norm = position / np.array(self.grid_size)
        vel_norm = velocity / 2.0 if np.linalg.norm(velocity) > 0 else np.zeros(2)
        wind_norm = wind / 5.0 if np.linalg.norm(wind) > 0 else np.zeros(2)
        
        # Créer le vecteur d'entrée
        input_vec = np.concatenate([pos_norm, vel_norm, wind_norm])
        
        # Propager à travers le réseau
        value = self.nn_forward(input_vec, self.state_eval_model)[0]
        
        # Bonus/malus basés sur la physique de la voile
        # Proximité à l'objectif
        dist_to_goal = np.linalg.norm(position - self.goal_position)
        goal_factor = np.exp(-dist_to_goal / 20.0)  # Décroissance exponentielle
        
        # Efficacité de navigation basée sur l'angle au vent
        if np.linalg.norm(wind) > 0.001 and np.linalg.norm(velocity) > 0.001:
            wind_from = -wind / np.linalg.norm(wind)
            boat_dir = velocity / np.linalg.norm(velocity)
            angle = np.arccos(np.clip(np.dot(wind_from, boat_dir), -1.0, 1.0))
            
            # Pénaliser les directions dans la zone interdite
            if angle < self.no_go_zone_angle:
                value -= 2.0
            # Favoriser le beam reach (perpendiculaire au vent)
            elif abs(angle - self.optimal_beam_reach_angle) < 0.2:
                value += 1.0
        
        # Intégrer le facteur de proximité au but
        value += 2.0 * goal_factor
        
        return value
    
    def predict_wind_evolution(self, current_wind, position=None, wind_field=None):
        """Prédit l'évolution du vent pour les prochains pas de temps."""
        # Si nous avons une mémoire de vent, l'utiliser pour la prédiction
        if len(self.wind_memory) > 5:
            # Aplatir la mémoire du vent
            wind_history = np.array(self.wind_memory).flatten()
            
            # Si la mémoire n'est pas complète, remplir avec la valeur actuelle
            if len(wind_history) < 20:
                padding = np.tile(current_wind, 10 - len(self.wind_memory))
                wind_history = np.concatenate([padding, wind_history])
            
            # Normaliser l'entrée
            wind_history_norm = wind_history / 5.0
            
            # Prédire avec le réseau de neurones
            predicted_wind = self.nn_forward(wind_history_norm, self.wind_prediction_model)
            
            # Dénormaliser la sortie
            predicted_wind = predicted_wind * 5.0
            
            return predicted_wind
        
        # Si pas assez d'historique, utiliser une heuristique basée sur le champ de vent actuel
        return current_wind * 0.98 + self.np_random.normal(0, 0.05, 2)
    
    def calculate_sailing_efficiency(self, boat_direction, wind_direction):
        """
        Calcule l'efficacité de navigation basée sur l'angle entre la direction du bateau et le vent.
        
        Args:
            boat_direction: Vecteur normalisé de la direction souhaitée du bateau
            wind_direction: Vecteur normalisé de la direction du vent (où le vent va VERS)
            
        Returns:
            sailing_efficiency: Flottant entre 0.05 et 1.0 représentant l'efficacité de navigation
        """
        # Inverser la direction du vent pour obtenir d'où vient le vent
        wind_from = -wind_direction
        
        # Calculer l'angle entre le vent et la direction
        cos_angle = np.dot(wind_from, boat_direction)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Éviter les erreurs numériques
        wind_angle = np.arccos(cos_angle)
        
        # Calcul de l'efficacité de navigation basé sur l'angle au vent
        if wind_angle < self.no_go_zone_angle:  # Moins de 30 degrés par rapport au vent
            sailing_efficiency = 0.05  # Efficacité faible mais non nulle dans la zone interdite
        elif wind_angle < self.close_hauled_angle:  # Entre 30 et 45 degrés
            sailing_efficiency = 0.1 + 0.4 * (wind_angle - self.no_go_zone_angle) / (self.close_hauled_angle - self.no_go_zone_angle)
        elif wind_angle < self.optimal_beam_reach_angle:  # Entre 45 et 90 degrés
            sailing_efficiency = 0.5 + 0.5 * (wind_angle - self.close_hauled_angle) / (self.optimal_beam_reach_angle - self.close_hauled_angle)
        elif wind_angle < 3*np.pi/4:  # Entre 90 et 135 degrés
            sailing_efficiency = 1.0  # Efficacité maximale
        else:  # Plus de 135 degrés
            sailing_efficiency = 1.0 - 0.2 * (wind_angle - 3*np.pi/4) / (np.pi/4)
            sailing_efficiency = max(0.8, sailing_efficiency)  # Mais toujours bien
        
        return sailing_efficiency
    
    def action_to_direction(self, action):
        """Convertit l'indice d'action en vecteur de direction."""
        directions = [
            (0, 1),     # 0: Nord
            (1, 1),     # 1: Nord-Est
            (1, 0),     # 2: Est
            (1, -1),    # 3: Sud-Est
            (0, -1),    # 4: Sud
            (-1, -1),   # 5: Sud-Ouest
            (-1, 0),    # 6: Ouest
            (-1, 1),    # 7: Nord-Ouest
            (0, 0)      # 8: Rester en place
        ]
        return np.array(directions[action])
    
    def direction_to_action(self, direction):
        """Convertit un vecteur de direction en indice d'action le plus proche."""
        directions = [
            (0, 1),     # 0: Nord
            (1, 1),     # 1: Nord-Est
            (1, 0),     # 2: Est
            (1, -1),    # 3: Sud-Est
            (0, -1),    # 4: Sud
            (-1, -1),   # 5: Sud-Ouest
            (-1, 0),    # 6: Ouest
            (-1, 1),    # 7: Nord-Ouest
            (0, 0)      # 8: Rester en place
        ]
        
        best_match = 8  # Par défaut: rester en place
        best_similarity = -np.inf
        
        # Si la direction est très petite, rester en place
        if np.linalg.norm(direction) < 0.01:
            return 8
        
        # Normaliser la direction
        direction = direction / np.linalg.norm(direction)
        
        # Trouver l'action la plus similaire à la direction
        for i, dir_vec in enumerate(directions[:8]):  # Exclure l'action "rester en place"
            dir_vec = np.array(dir_vec)
            dir_norm = dir_vec / np.linalg.norm(dir_vec)
            similarity = np.dot(direction, dir_norm)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = i
                
        return best_match
    
    def analyze_wind_field(self, wind_field_flat, position):
        """
        Analyse sophistiquée du champ de vent pour identifier des structures et modèles.
        Implémente un clustering et une détection de corridors de vent.
        """
        # Reconstruire le champ de vent 2D à partir des données aplaties
        wind_field = wind_field_flat.reshape(self.grid_size[1], self.grid_size[0], 2)
        
        # Calculer le vecteur vers l'objectif
        goal_vector = self.goal_position - position
        goal_distance = np.linalg.norm(goal_vector)
        if goal_distance < 0.001:
            return 8, None  # Déjà à l'objectif
            
        goal_direction = goal_vector / goal_distance
        
        # 1. Recherche de corridors de vent favorable
        # Échantillonner des points le long de différents chemins potentiels
        num_paths = 7  # Nombre de chemins à tester
        path_angles = np.linspace(-np.pi/3, np.pi/3, num_paths)  # Angles relatifs à la direction de l'objectif
        path_scores = np.zeros(num_paths)
        path_efficiencies = []
        
        for p_idx, angle in enumerate(path_angles):
            # Calculer la direction du chemin
            path_direction = np.array([
                goal_direction[0] * np.cos(angle) - goal_direction[1] * np.sin(angle),
                goal_direction[0] * np.sin(angle) + goal_direction[1] * np.cos(angle)
            ])
            
            # Échantillonner des points le long du chemin
            path_positions = []
            path_winds = []
            
            for step in range(1, 10):  # 10 points par chemin
                # Position le long du chemin
                step_pos = position + path_direction * step * 2  # Pas de 2 unités
                x, y = int(np.clip(step_pos[0], 0, self.grid_size[0]-1)), int(np.clip(step_pos[1], 0, self.grid_size[1]-1))
                
                path_positions.append((x, y))
                path_winds.append(wind_field[y, x])
            
            # Calculer l'efficacité moyenne le long du chemin
            efficiency_scores = []
            for wind_vec in path_winds:
                if np.linalg.norm(wind_vec) > 0.001:
                    wind_normalized = wind_vec / np.linalg.norm(wind_vec)
                    efficiency = self.calculate_sailing_efficiency(path_direction, wind_normalized)
                    efficiency_scores.append(efficiency)
            
            # Score du chemin = efficacité moyenne * proximité à l'objectif
            if efficiency_scores:
                mean_efficiency = np.mean(efficiency_scores)
                # Facteur de direction vers l'objectif (plus élevé si le chemin mène vers l'objectif)
                goal_alignment = np.dot(path_direction, goal_direction)
                path_scores[p_idx] = mean_efficiency * (0.5 + 0.5 * goal_alignment)
                path_efficiencies.append((path_direction, mean_efficiency, path_scores[p_idx]))
        
        # 2. Recherche de gradients et tendances dans le champ de vent
        # Calculer les dérivées spatiales du champ de vent
        grad_x = np.zeros_like(wind_field)
        grad_y = np.zeros_like(wind_field)
        
        # Calcul simple des gradients (différences de premier ordre)
        for i in range(1, wind_field.shape[0]-1):
            for j in range(1, wind_field.shape[1]-1):
                grad_x[i, j] = (wind_field[i, j+1] - wind_field[i, j-1]) / 2
                grad_y[i, j] = (wind_field[i+1, j] - wind_field[i-1, j]) / 2
        
        # Calculer la divergence et le rotationnel pour identifier les structures de vent
        divergence = np.zeros((wind_field.shape[0], wind_field.shape[1]))
        curl = np.zeros((wind_field.shape[0], wind_field.shape[1]))
        
        for i in range(1, wind_field.shape[0]-1):
            for j in range(1, wind_field.shape[1]-1):
                divergence[i, j] = grad_x[i, j, 0] + grad_y[i, j, 1]
                curl[i, j] = grad_x[i, j, 1] - grad_y[i, j, 0]
        
        # Identifier les zones de haute pression (divergence positive) et basse pression (divergence négative)
        # Ces zones sont importantes en navigation car elles indiquent des changements potentiels de vent
        x, y = int(position[0]), int(position[1])
        local_divergence = 0
        local_curl = 0
        
        if 0 <= x < divergence.shape[1] and 0 <= y < divergence.shape[0]:
            local_divergence = divergence[y, x]
            local_curl = curl[y, x]
        
        # 3. Synthèse de l'analyse et décision
        # Trouver le meilleur chemin basé sur les scores calculés
        best_path_idx = np.argmax(path_scores)
        best_path_direction = np.array([
            goal_direction[0] * np.cos(path_angles[best_path_idx]) - goal_direction[1] * np.sin(path_angles[best_path_idx]),
            goal_direction[0] * np.sin(path_angles[best_path_idx]) + goal_direction[1] * np.cos(path_angles[best_path_idx])
        ])
        
        # Convertir en action
        best_action = self.direction_to_action(best_path_direction)
        
        # Si toutes les efficacités sont très basses, envisager une stratégie de louvoyage
        if np.max(path_scores) < 0.3:
            # Situation probable de vent contraire, utiliser le louvoyage
            return self.get_upwind_tacking_action(position, wind_field[y, x]), path_efficiencies
        
        return best_action, path_efficiencies
    
    def get_upwind_tacking_action(self, position, wind_vector):
        """
        Stratégie avancée de louvoyage pour naviguer contre le vent.
        Alterne entre les directions à environ 45 degrés de part et d'autre du vent.
        """
        # Normaliser le vecteur de vent
        wind_magnitude = np.linalg.norm(wind_vector)
        if wind_magnitude < 0.001:
            return 0  # Par défaut, aller au nord si pas de vent
            
        wind_normalized = wind_vector / wind_magnitude
        
        # Calcul de l'angle du vent (d'où il vient)
        wind_from_angle = np.arctan2(-wind_normalized[1], -wind_normalized[0])
        
        # Calculer les angles de louvoyage (environ 45 degrés de chaque côté du vent)
        port_tack_angle = wind_from_angle - self.close_hauled_angle
        starboard_tack_angle = wind_from_angle + self.close_hauled_angle
        
        # Décider de changer de bord si nécessaire
        distance_to_center = abs(position[0] - self.grid_size[0]/2)
        
        # Changer de bord si on s'écarte trop du centre, ou si on est bloqué
        if distance_to_center > self.grid_size[0]/4 or self.steps_without_progress > 5:
            self.current_tack_direction *= -1
            self.steps_without_progress = 0
        
        # Sélectionner l'angle en fonction du bord actuel
        tack_angle = starboard_tack_angle if self.current_tack_direction > 0 else port_tack_angle
        
        # Convertir l'angle en vecteur de direction
        tack_direction = np.array([np.cos(tack_angle), np.sin(tack_angle)])
        
        # Convertir la direction en action
        return self.direction_to_action(tack_direction)
    
    def a_star_search(self, start_pos, goal_pos, wind_field):
        """
        Algorithme A* adapté à la navigation à voile.
        Tient compte de l'efficacité de navigation basée sur le vent.
        """
        # Clé de cache pour cette recherche
        cache_key = (tuple(start_pos), tuple(goal_pos))
        
        # Vérifier si ce chemin est déjà dans le cache
        if cache_key in self.path_cache:
            # Vérifier si le cache est encore valide (moins de X étapes)
            if self.steps_since_path_recompute < self.path_recompute_threshold:
                return self.path_cache[cache_key]
        
        # Fonctions d'aide pour A*
        def heuristic(pos, goal):
            return np.linalg.norm(np.array(pos) - np.array(goal))
        
        def get_neighbors(pos):
            neighbors = []
            for action in range(8):  # Toutes les directions sauf "rester en place"
                direction = self.action_to_direction(action)
                next_pos = (int(pos[0] + direction[0]), int(pos[1] + direction[1]))
                
                # Vérifier les limites
                if 0 <= next_pos[0] < self.grid_size[0] and 0 <= next_pos[1] < self.grid_size[1]:
                    neighbors.append((next_pos, action))
            return neighbors
        
        def edge_cost(current, next_pos, action, wind_field):
            # Direction du mouvement
            direction = self.action_to_direction(action)
            direction_norm = direction / np.linalg.norm(direction)
            
            # Vent à la position actuelle
            current_wind = wind_field[int(current[1]), int(current[0])]
            
            # Échantillonner le vent à mi-chemin entre les positions pour plus de précision
            mid_x = int((current[0] + next_pos[0]) / 2)
            mid_y = int((current[1] + next_pos[1]) / 2)
            mid_wind = wind_field[mid_y, mid_x]
            
            # Moyenne des vents
            avg_wind = (current_wind + mid_wind) / 2
            
            if np.linalg.norm(avg_wind) < 0.001:
                return 2.0  # Coût par défaut si pas de vent
            
            # Normaliser le vent
            wind_norm = avg_wind / np.linalg.norm(avg_wind)
            
            # Calculer l'efficacité de navigation
            efficiency = self.calculate_sailing_efficiency(direction_norm, wind_norm)
            
            # Le coût est inversement proportionnel à l'efficacité
            # Plus l'efficacité est élevée, plus le coût est faible
            cost = 1.0 / (efficiency + 0.1)
            
            # Pénalité pour la zone interdite (contre le vent)
            wind_from = -wind_norm
            angle = np.arccos(np.clip(np.dot(wind_from, direction_norm), -1.0, 1.0))
            if angle < self.no_go_zone_angle:
                cost *= 5.0  # Forte pénalité
            
            return cost
        
        # Initialisation de l'algorithme A*
        start = tuple(map(int, start_pos))
        goal = tuple(map(int, goal_pos))
        
        # Liste ouverte (nœuds à explorer) et liste fermée (nœuds déjà explorés)
        open_set = []
        closed_set = set()
        
        # Dictionnaires pour stocker les scores g et f, et les parents
        g_score = defaultdict(lambda: float('inf'))
        f_score = defaultdict(lambda: float('inf'))
        parent = {}
        actions = {}
        
        # Initialiser le nœud de départ
        g_score[start] = 0
        f_score[start] = heuristic(start, goal)
        heapq.heappush(open_set, (f_score[start], start))
        
        # Boucle principale de A*
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                # Reconstruire le chemin
                path = []
                action_path = []
                while current in parent:
                    path.append(current)
                    action_path.append(actions[current])
                    current = parent[current]
                
                path.reverse()
                action_path.reverse()
                
                # Mettre en cache le chemin
                self.path_cache[cache_key] = action_path
                self.steps_since_path_recompute = 0
                
                return action_path
            
            closed_set.add(current)
            
            # Explorer les voisins
            for neighbor, action in get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                # Calculer le nouveau score g
                tentative_g = g_score[current] + edge_cost(current, neighbor, action, wind_field)
                
                if tentative_g < g_score[neighbor]:
                    # Ce chemin vers le voisin est meilleur
                    parent[neighbor] = current
                    actions[neighbor] = action
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                    
                    # Ajouter à la liste ouverte si pas déjà dedans
                    if neighbor not in [i[1] for i in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # Si aucun chemin n'est trouvé, retourner une action par défaut
        # Essayer d'aller directement vers l'objectif
        return [self.direction_to_action(goal_pos - start_pos)]
    
    def act(self, observation):
        """
        Méthode principale de sélection d'action.
        Combine A*, analyse de vent et évaluation par réseau de neurones.
        """
        # Extraire les informations de l'observation
        position = np.array([observation[0], observation[1]])
        velocity = np.array([observation[2], observation[3]])
        wind_at_position = np.array([observation[4], observation[5]])
        wind_field_flat = observation[6:]
        
        # Stocker le vent actuel dans la mémoire
        self.wind_memory.append(wind_at_position)
        
        # Reconstruire le champ de vent 2D
        wind_field = wind_field_flat.reshape(self.grid_size[1], self.grid_size[0], 2)
        
        # Calculer le vecteur vers l'objectif
        goal_vector = self.goal_position - position
        current_distance_to_goal = np.linalg.norm(goal_vector)
        
        # Suivre les progrès
        if self.previous_position is not None:
            distance_improvement = self.last_distance_to_goal - current_distance_to_goal
            if distance_improvement < 0.1:  # Si on ne progresse pas assez
                self.steps_without_progress += 1
            else:
                self.steps_without_progress = 0
        
        self.previous_position = position.copy()
        self.last_distance_to_goal = current_distance_to_goal
        self.steps_since_path_recompute += 1

        # Exploration aléatoire avec faible probabilité
        if self.np_random.random() < self.exploration_rate:
            return self.np_random.integers(0, 9)
        
        # Combiner les différentes approches pour sélectionner l'action
        
        # 1. Planification de trajectoire avec A*
        if self.use_astar and (not self.current_path or self.steps_since_path_recompute >= self.path_recompute_threshold):
            self.current_path = self.a_star_search(position, self.goal_position, wind_field)
        
        # 2. Analyse du champ de vent pour identifier les corridors favorables
        wind_analysis_action, path_efficiencies = self.analyze_wind_field(wind_field_flat, position)
        
        # 3. Évaluation des états possibles avec le réseau de neurones
        nn_evaluations = []
        for action in range(8):  # Évaluer toutes les actions possibles
            direction = self.action_to_direction(action)
            next_pos = position + direction
            
            # Vérifier les limites
            if (0 <= next_pos[0] < self.grid_size[0] and 
                0 <= next_pos[1] < self.grid_size[1]):
                
                # Prédire l'évolution du vent
                future_wind = self.predict_wind_evolution(wind_at_position)
                
                # Estimer la nouvelle vitesse (simplifié)
                boat_dir = direction / np.linalg.norm(direction)
                wind_norm = wind_at_position / (np.linalg.norm(wind_at_position) + 1e-10)
                sailing_efficiency = self.calculate_sailing_efficiency(boat_dir, wind_norm)
                next_vel = boat_dir * sailing_efficiency * np.linalg.norm(wind_at_position) * 0.4  # 0.4 ~ boat_performance
                
                # Évaluer cet état potentiel avec le réseau de neurones
                state_value = self.evaluate_state(next_pos, next_vel, future_wind)
                
                nn_evaluations.append((action, state_value))
        
        # Trier les évaluations par valeur décroissante
        nn_evaluations.sort(key=lambda x: x[1], reverse=True)
        
        # Si nous avons un chemin A* valide, l'utiliser prioritairement sauf si bloqué
        if self.current_path and self.steps_without_progress < 5:
            if len(self.current_path) > 0:
                astar_action = self.current_path[0]
                self.current_path = self.current_path[1:] if len(self.current_path) > 1 else []
                return astar_action
        
        # Si nous sommes bloqués ou n'avons pas de chemin A*, utiliser l'analyse de vent
        if self.steps_without_progress >= 5 or not self.current_path:
            # Si nous sommes vraiment bloqués, essayer de nous libérer en utilisant le louvoyage
            if self.steps_without_progress >= 10:
                # Alterner entre les bords pour essayer de sortir du blocage
                self.current_tack_direction *= -1
                return self.get_upwind_tacking_action(position, wind_at_position)
            
            # Utiliser l'analyse de vent pour naviguer
            return wind_analysis_action
        
        # En dernier recours, utiliser l'évaluation du réseau de neurones
        if nn_evaluations:
            return nn_evaluations[0][0]
        
        # Si tout échoue, essayer d'aller directement vers l'objectif
        goal_direction = goal_vector / (np.linalg.norm(goal_vector) + 1e-10)
        return self.direction_to_action(goal_direction)