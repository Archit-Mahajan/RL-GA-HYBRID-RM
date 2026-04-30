import AsyncStorage from '@react-native-async-storage/async-storage';
import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';

export type Intake = {
  id: string;
  amount_ml: number;
  timestamp: string;
  cup_size: number;
};

export type DailyLog = {
  date: string;
  intakes: Intake[];
  goal_ml: number;
  streak: number;
};

export type ActivityLevel = 'low' | 'medium' | 'high';
export type Climate = 'cold' | 'temperate' | 'hot';

export type UserProfile = {
  weight_kg: number;
  activity_level: ActivityLevel;
  climate: Climate;
  custom_cup_sizes: number[];
  reminder_times: string[];
};

export type AppState = {
  currentStreak: number;
  longestStreak: number;
  weeklyData: DailyLog[];
  monthlyData: DailyLog[];
};

export type WaterTrackerState = {
  dailyLog: DailyLog;
  userProfile: UserProfile;
  appState: AppState;
  addIntake: (amount_ml: number, cup_size: number, timestamp?: string) => void;
  removeIntake: (id: string) => void;
  setGoal: (goal_ml: number) => void;
  resetDay: () => void;
  updateStreak: () => void;
  ensureTodayLog: () => void;
};

const DEFAULT_GOAL_ML = 2000;

const createIntakeId = () =>
  `${Date.now()}-${Math.random().toString(16).slice(2, 10)}`;

const getDateKey = (date: Date = new Date()) =>
  date.toISOString().split('T')[0];

const dateKeyToMidnight = (dateKey: string) =>
  new Date(`${dateKey}T00:00:00.000Z`);

const isNextDay = (previous: string, next: string) =>
  dateKeyToMidnight(next).getTime() - dateKeyToMidnight(previous).getTime() ===
  24 * 60 * 60 * 1000;

const sumIntakes = (log: DailyLog) =>
  log.intakes.reduce((total, intake) => total + intake.amount_ml, 0);

const metGoal = (log: DailyLog) =>
  log.goal_ml > 0 && sumIntakes(log) >= log.goal_ml;

const sortLogs = (logs: DailyLog[]) =>
  [...logs].sort((a, b) => a.date.localeCompare(b.date));

const trimLogs = (logs: DailyLog[], maxDays: number) => {
  const sorted = sortLogs(logs);
  return sorted.slice(Math.max(0, sorted.length - maxDays));
};

const computeStreakStats = (logs: DailyLog[]) => {
  const sorted = sortLogs(logs);
  let longest = 0;
  let currentRun = 0;

  for (let i = 0; i < sorted.length; i += 1) {
    const log = sorted[i];
    const previousLog = sorted[i - 1];

    if (!metGoal(log)) {
      currentRun = 0;
      continue;
    }

    if (i === 0 || !previousLog || !metGoal(previousLog) || !isNextDay(previousLog.date, log.date)) {
      currentRun = 1;
    } else {
      currentRun += 1;
    }

    if (currentRun > longest) {
      longest = currentRun;
    }
  }

  let current = 0;
  if (sorted.length > 0) {
    const lastIndex = sorted.length - 1;
    const lastLog = sorted[lastIndex];
    if (metGoal(lastLog)) {
      current = 1;
      for (let i = lastIndex; i > 0; i -= 1) {
        const prev = sorted[i - 1];
        const next = sorted[i];
        if (!metGoal(prev) || !isNextDay(prev.date, next.date)) {
          break;
        }
        current += 1;
      }
    }
  }

  return { current, longest };
};

const createDailyLog = (date: string, goal_ml: number, streak: number): DailyLog => ({
  date,
  intakes: [],
  goal_ml,
  streak,
});

const defaultProfile: UserProfile = {
  weight_kg: 70,
  activity_level: 'medium',
  climate: 'temperate',
  custom_cup_sizes: [250, 500],
  reminder_times: [],
};

export const useWaterTrackerStore = create<WaterTrackerState>()(
  persist(
    (set, get) => ({
      dailyLog: createDailyLog(getDateKey(), DEFAULT_GOAL_ML, 0),
      userProfile: defaultProfile,
      appState: {
        currentStreak: 0,
        longestStreak: 0,
        weeklyData: [],
        monthlyData: [],
      },
      addIntake: (amount_ml, cup_size, timestamp = new Date().toISOString()) => {
        set((state) => ({
          dailyLog: {
            ...state.dailyLog,
            intakes: [
              ...state.dailyLog.intakes,
              { id: createIntakeId(), amount_ml, timestamp, cup_size },
            ],
          },
        }));
        get().updateStreak();
      },
      removeIntake: (id) => {
        set((state) => ({
          dailyLog: {
            ...state.dailyLog,
            intakes: state.dailyLog.intakes.filter((intake) => intake.id !== id),
          },
        }));
        get().updateStreak();
      },
      setGoal: (goal_ml) => {
        set((state) => ({
          dailyLog: {
            ...state.dailyLog,
            goal_ml,
          },
        }));
        get().updateStreak();
      },
      resetDay: () => {
        set((state) => {
          const today = getDateKey();
          if (state.dailyLog.date === today) {
            return state;
          }

          const previousLog = state.dailyLog;
          const weeklyData = trimLogs(
            [...state.appState.weeklyData, previousLog],
            7,
          );
          const monthlyData = trimLogs(
            [...state.appState.monthlyData, previousLog],
            30,
          );
          const streakStats = computeStreakStats(monthlyData);
          const longestStreak = Math.max(
            state.appState.longestStreak,
            streakStats.longest,
          );

          return {
            dailyLog: createDailyLog(
              today,
              previousLog.goal_ml,
              streakStats.current,
            ),
            appState: {
              ...state.appState,
              currentStreak: streakStats.current,
              longestStreak,
              weeklyData,
              monthlyData,
            },
          };
        });
      },
      updateStreak: () => {
        set((state) => {
          const logs = trimLogs(
            [...state.appState.monthlyData, state.dailyLog],
            30,
          );
          const streakStats = computeStreakStats(logs);
          const longestStreak = Math.max(
            state.appState.longestStreak,
            streakStats.longest,
          );

          return {
            dailyLog: {
              ...state.dailyLog,
              streak: streakStats.current,
            },
            appState: {
              ...state.appState,
              currentStreak: streakStats.current,
              longestStreak,
            },
          };
        });
      },
      ensureTodayLog: () => {
        const today = getDateKey();
        const { dailyLog } = get();
        if (dailyLog.date !== today) {
          get().resetDay();
        }
      },
    }),
    {
      name: 'water-tracker-store',
      storage: createJSONStorage(() => AsyncStorage),
      onRehydrateStorage: () => (state) => {
        state?.ensureTodayLog();
      },
    },
  ),
);

export const todayTotal = (state: WaterTrackerState) => sumIntakes(state.dailyLog);

export const todayProgress = (state: WaterTrackerState) => {
  const goal = state.dailyLog.goal_ml;
  if (goal <= 0) {
    return 0;
  }
  return Math.min(1, todayTotal(state) / goal);
};

export const currentStreak = (state: WaterTrackerState) =>
  state.appState.currentStreak;
