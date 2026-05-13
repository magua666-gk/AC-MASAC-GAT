"""
课程学习包 - 用于MASAC算法的课程学习框架

此包提供了一套完整的工具，用于实现基于课程的强化学习，
包括任务生成、任务排序、知识迁移和性能评估等功能。
"""

from .task import Task
from .task_generator import TaskGenerator, DefaultTaskGenerator, ProgressiveTaskGenerator,FixedTaskGenerator
from .task_sequencer import TaskSequencer, LinearTaskSequencer, LearningProgressTaskSequencer, AdaptiveTaskSequencer
from .knowledge_transfer import KnowledgeTransfer, PolicyTransfer, ValueFunctionTransfer, HybridTransfer
from .curriculum_manager import CurriculumManager
from .utils.config import CurriculumConfig

__version__ = '0.1.0'

__all__ = [
    'Task',
    'TaskGenerator',
    'DefaultTaskGenerator',
    'ProgressiveTaskGenerator',
    'TaskSequencer',
    'LinearTaskSequencer',
    'LearningProgressTaskSequencer',
    'AdaptiveTaskSequencer',
    'KnowledgeTransfer',
    'PolicyTransfer',
    'ValueFunctionTransfer',
    'HybridTransfer',
    'CurriculumManager',
    'CurriculumConfig'
] 