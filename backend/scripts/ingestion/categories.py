# backend/scripts/ingestion/categories.py

"""
Comprehensive list of arXiv categories for Computer Science and Engineering
"""

# All Computer Science categories
CS_CATEGORIES = [
    # Core AI/ML
    'cs.AI',      # Artificial Intelligence
    'cs.CL',      # Computation and Language
    'cs.LG',      # Machine Learning
    'cs.CV',      # Computer Vision and Pattern Recognition
    'cs.NE',      # Neural and Evolutionary Computing
    'cs.RO',      # Robotics
    
    # Systems and Architecture
    'cs.AR',      # Hardware Architecture
    'cs.OS',      # Operating Systems
    'cs.DC',      # Distributed, Parallel, and Cluster Computing
    'cs.SE',      # Software Engineering
    'cs.PL',      # Programming Languages
    'cs.DS',      # Data Structures and Algorithms
    
    # Security and Networks
    'cs.CR',      # Cryptography and Security
    'cs.NI',      # Networking and Internet Architecture
    'cs.IT',      # Information Theory
    
    # Data and Databases
    'cs.DB',      # Databases
    'cs.DL',      # Digital Libraries
    'cs.IR',      # Information Retrieval
    'cs.DM',      # Discrete Mathematics
    
    # Graphics and Multimedia
    'cs.GR',      # Graphics
    'cs.MM',      # Multimedia
    'cs.CG',      # Computational Geometry
    
    # Human-Computer Interaction
    'cs.HC',      # Human-Computer Interaction
    
    # Theory
    'cs.CC',      # Computational Complexity
    'cs.LO',      # Logic in Computer Science
    'cs.ET',      # Emerging Technologies
    'cs.FL',      # Formal Languages and Automata Theory
    'cs.GT',      # Computer Science and Game Theory
    'cs.SC',      # Symbolic Computation
    
    # Software and Systems
    'cs.SI',      # Social and Information Networks
    'cs.SY',      # Systems and Control
    'cs.SD',      # Sound
    'cs.CE',      # Computational Engineering, Finance, and Science
    'cs.MS',      # Mathematical Software
    'cs.NA',      # Numerical Analysis
    'cs.PF',      # Performance
    'cs.CY',      # Computers and Society
    'cs.OH',      # Other Computer Science
]

# Electrical Engineering and Systems Science
EESS_CATEGORIES = [
    'eess.AS',    # Audio and Speech Processing
    'eess.IV',    # Image and Video Processing
    'eess.SP',    # Signal Processing
    'eess.SY',    # Systems and Control
]

# Mathematics (relevant to CS)
MATH_CATEGORIES = [
    'math.OC',    # Optimization and Control
    'math.NA',    # Numerical Analysis
    'math.ST',    # Statistics Theory
    'math.IT',    # Information Theory
    'math.CO',    # Combinatorics
    'math.DS',    # Dynamical Systems
]

# Statistics (relevant to ML/CS)
STAT_CATEGORIES = [
    'stat.ML',    # Machine Learning
    'stat.AP',    # Applications
    'stat.TH',    # Statistics Theory
    'stat.CO',    # Computation
]

# Quantitative Biology (Biotech/Bioinformatics)
QBIO_CATEGORIES = [
    'q-bio.BM',   # Biomolecules
    'q-bio.CB',   # Cell Behavior
    'q-bio.GN',   # Genomics
    'q-bio.MN',   # Molecular Networks
    'q-bio.NC',   # Neurons and Cognition (AI/neuroscience intersection)
    'q-bio.OT',   # Other Quantitative Biology
    'q-bio.PE',   # Populations and Evolution
    'q-bio.QM',   # Quantitative Methods (computational biology)
    'q-bio.SC',   # Subcellular Processes
    'q-bio.TO',   # Tissues and Organs
]

# Computational Finance (AI/ML applications)
QFIN_CATEGORIES = [
    'q-fin.CP',   # Computational Finance
    'q-fin.ST',   # Statistical Finance
    'q-fin.TR',   # Trading and Market Microstructure
]

# Economics (AI/ML applications)
ECON_CATEGORIES = [
    'econ.EM',    # Econometrics (data science)
    'econ.TH',    # Theoretical Economics (game theory)
    'econ.GN',    # General Economics
]

# Physics (Data Analysis and Computational)
PHYSICS_CATEGORIES = [
    'physics.data-an',  # Data Analysis, Statistics and Probability
    'physics.soc-ph',   # Physics and Society (network science)
    'physics.comp-ph',  # Computational Physics
]

# All categories combined
ALL_CS_AND_ENG_CATEGORIES = (
    CS_CATEGORIES + 
    EESS_CATEGORIES + 
    MATH_CATEGORIES + 
    STAT_CATEGORIES +
    QBIO_CATEGORIES +
    QFIN_CATEGORIES +
    ECON_CATEGORIES +
    PHYSICS_CATEGORIES
)

# Preset category groups
PRESETS = {
    'cs-core': [
        'cs.AI', 'cs.CL', 'cs.LG', 'cs.CV', 'cs.NE', 'cs.RO'
    ],
    'cs-systems': [
        'cs.OS', 'cs.DC', 'cs.SE', 'cs.PL', 'cs.AR', 'cs.DS'
    ],
    'cs-data': [
        'cs.DB', 'cs.IR', 'cs.DL', 'cs.DM', 'cs.IT'
    ],
    'cs-security': [
        'cs.CR', 'cs.NI', 'cs.SY'
    ],
    'cs-graphics': [
        'cs.GR', 'cs.MM', 'cs.CG', 'cs.HC'
    ],
    'cs-all': CS_CATEGORIES,
    'eess-all': EESS_CATEGORIES,
    'math-relevant': MATH_CATEGORIES,
    'stat-relevant': STAT_CATEGORIES,
    'biotech': QBIO_CATEGORIES,
    'quant-finance': QFIN_CATEGORIES,
    'ai-applications': (
        STAT_CATEGORIES + 
        QFIN_CATEGORIES + 
        ECON_CATEGORIES + 
        PHYSICS_CATEGORIES
    ),
    'comprehensive': ALL_CS_AND_ENG_CATEGORIES,
}

