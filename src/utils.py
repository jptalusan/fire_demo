import os

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import pandas as pd

def preprocess(df, cat_columns, reg_columns):
    # Copy the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Ensure 'AlarmDate' column is datetime
    if 'AlarmDate' not in df_copy.columns:
        raise ValueError("'AlarmDate' column is missing from the DataFrame.")
    df_copy['AlarmDate'] = pd.to_datetime(df_copy['AlarmDate'])

    # Step 1: One-hot encode the categorical columns
    df_encoded = pd.get_dummies(df_copy, columns=cat_columns, drop_first=True)
    for col in reg_columns:
        df_encoded['original_'+col] = df_encoded[col]
        
        
    
    # Split the DataFrame into training and testing based on the 'AlarmDate' column
    train_df = df_encoded[df_encoded['AlarmDate'] < pd.to_datetime('2023-11-01')].copy()
    test_df = df_encoded[df_encoded['AlarmDate'] >= pd.to_datetime('2023-11-01')].copy()
    
    # Step 2: Normalize the regression columns (numerical columns)
    scaler = StandardScaler()
    if len(reg_columns) > 0:

        
        train_df[reg_columns] = scaler.fit_transform(train_df[reg_columns])
        test_df[reg_columns] = scaler.transform(test_df[reg_columns])

    # Step 3: Determine the feature columns
    # Exclude non-feature columns (like 'AlarmDate') to identify the feature set
    encoded_cat_columns = [col for col in df_encoded.columns if (col not in df.columns or col in cat_columns) & (not col.startswith('original_'))]
    features = reg_columns + encoded_cat_columns
    
    return train_df, test_df, features, scaler


def solve_pmedian_with_balance(E, L, a, d, p, alpha=0):
    """
    Solve:  min_{X,Y}  sum_{i∈E,j∈L} a[i]*d[i,j]*Y[i,j]*b[j]
    s.t.   sum_j Y[i,j] == 1      ∀ i∈E
           sum_j X[j] == p
           Y[i,j] <= X[j]         ∀ i,j
           X[j], Y[i,j] ∈ {0,1}
    where for alpha=0    b[j]=1
          for alpha=1    b[j] = ( sum_i a[i]*Y[i,j] ) / A
    """

    A = sum(a[i] for i in E)       # total demand
    model = gp.Model("p_median_balance")

    # Variables
    X = model.addVars(L, vtype=GRB.BINARY, name="X")
    Y = model.addVars(E, L, vtype=GRB.BINARY, name="Y")

    if alpha == 1:
        # t[j] = sum_i a[i] * Y[i,j]
        t = model.addVars(L, lb=0.0, name="t")

    # Constraints
    # 1) each demand point i must be assigned to exactly one facility
    for i in E:
        model.addConstr(Y.sum(i, "*") == 1, name=f"assign_{i}")

    # 2) exactly p facilities opened
    model.addConstr(X.sum() == p, name="num_facilities")

    # 3) assignment only if facility is open
    for i in E:
        for j in L:
            model.addConstr(Y[i, j] <= X[j], name=f"link_{i}_{j}")

    if alpha == 1:
        # define t[j]
        for j in L:
            model.addConstr(
                t[j] == gp.quicksum(a[i] * Y[i, j] for i in E),
                name=f"t_def_{j}"
            )

    # Objective
    if alpha == 0:
        # classical p-median
        obj = gp.quicksum(a[i] * d[i, j] * Y[i, j]
                          for i in E for j in L)
        model.setObjective(obj, GRB.MINIMIZE)

    elif alpha == 1:
        # nonconvex quadratic: sum_{i,j} a[i]*d[i,j]*Y[i,j]*(t[j]/A)
        expr = gp.QuadExpr()
        for i in E:
            for j in L:
                expr.add(a[i] * d[i, j] * Y[i, j] * (t[j] / A))
        model.setObjective(expr, GRB.MINIMIZE)
        # enable non-convex QP
        model.Params.NonConvex = 2

    else:
        raise NotImplementedError("Only α=0 or α=1 supported in this code")

    # Solve
    model.optimize()
    return model, X, Y

def add_p_via_mip(E, L, a, d, X_exist, p_add, alpha=0):
    p_new = len(X_exist) + p_add
    A = sum(a[i] for i in E)

    m = gp.Model()
    X = m.addVars(L, vtype=GRB.BINARY, name="X")
    Y = m.addVars(E, L, vtype=GRB.BINARY, name="Y")
    if alpha == 1:
        t = m.addVars(L, lb=0.0, name="t")

    # 1) fix existing
    for j in X_exist:
        m.addConstr(X[j] == 1)

    # 2) open exactly p_new
    m.addConstr(X.sum() == p_new)

    # 3) assignment + link
    for i in E:
        m.addConstr(Y.sum(i, "*") == 1)
        for j in L:
            m.addConstr(Y[i, j] <= X[j])

    # 4) balancing term
    if alpha == 1:
        for j in L:
            m.addConstr(t[j] == gp.quicksum(a[i]*Y[i, j] for i in E))

    # 5) objective
    if alpha == 0:
        m.setObjective(
            gp.quicksum(a[i]*d[i, j]*Y[i, j] for i in E for j in L),
            GRB.MINIMIZE
        )
    else:
        expr = gp.QuadExpr()
        for i in E:
            for j in L:
                expr.add(a[i]*d[i, j]*Y[i, j]*(t[j]/A))
        m.setObjective(expr, GRB.MINIMIZE)
        m.Params.NonConvex = 2

    m.optimize()
    return m, X, Y


import gurobipy as gp
from gurobipy import GRB
from collections import Counter

def add_p_via_mip_multi(E, L, a, d, fire_stations, p_add, alpha=0):
    """
    E: list of demand indices
    L: list of candidate-cell indices
    a: dict i->demand weight
    d: dict (i,j)->distance
    fire_stations: list of cell-indices (may contain duplicates)
    p_add: how many new facilities to add
    alpha: 0 (classic) or 1 (balanced)
    """
    # 1) Count how many existing stations in each cell
    counts = Counter(fire_stations)
    P0 = sum(counts.values())
    p_new = P0 + p_add
    A = sum(a[i] for i in E)

    m = gp.Model("p_median_multi")

    # 2) X[j] integer in [0..p_new]
    X = m.addVars(L, lb=0, ub=p_new, vtype=GRB.INTEGER, name="X")
    Y = m.addVars(E, L, vtype=GRB.BINARY, name="Y")

    # 3) b[j] ∈ [0,1] is the normalized load fraction (only if alpha=1)
    if alpha == 1:
        b = m.addVars(L, lb=0.0, ub=1.0, name="b")

    # 4) Fix existing counts:
    for j, cnt in counts.items():
        m.addConstr(X[j] == cnt, name=f"fix_exist_{j}")

    # 5) Total facilities = old + new
    m.addConstr(X.sum() == p_new, name="total_facilities")

    # 6) Assignment & linkage
    for i in E:
        m.addConstr(Y.sum(i, "*") == 1, name=f"assign_{i}")
        for j in L:
            m.addConstr(Y[i,j] <= X[j], name=f"link_{i}_{j}")

    # 7) Define b[j] = (sum_i a[i]*Y[i,j]) / A
    if alpha == 1:
        for j in L:
            m.addConstr(
                gp.quicksum(a[i]*Y[i,j] for i in E) == b[j] * A,
                name=f"norm_{j}"
            )

    # 8) Objective
    if alpha == 0:
        obj = gp.quicksum(a[i]*d[i,j]*Y[i,j] for i in E for j in L)
        m.setObjective(obj, GRB.MINIMIZE)
    else:
        # now use b[j] directly (in [0,1])
        expr = gp.quicksum(a[i]*d[i,j]*Y[i,j]*b[j] for i in E for j in L)
        m.setObjective(expr, GRB.MINIMIZE)

    m.Params.NonConvex = int(alpha>0)  # allow nonconvex if we used b
    m.optimize()

    return m, X, Y, (b if alpha==1 else None)


#!/usr/bin/env python3
"""
p_median_relocation.py

This script shows how to:
  1. Solve the (balanced) p-median problem.
  2. Add p new stations to an existing set via exact MIP or a greedy heuristic.
  3. Relocate p existing stations (close them and open p replacements) via exact MIP or greedy.

Requires Gurobi 12+ and gurobipy.
"""

import math
import gurobipy as gp
from gurobipy import GRB

def solve_pmedian_with_balance(E, L, a, d, p, alpha=0):
    """
    Solve the p-median (alpha=0) or balanced p-median (alpha=1) exactly.
    E: list of demand indices
    L: list of candidate site indices
    a: dict mapping i->demand weight
    d: dict mapping (i,j)->distance
    p: number of facilities to open
    alpha: 0 or 1 (only these are supported here)
    Returns: (model, X_vars, Y_vars)
    """
    A = sum(a[i] for i in E)
    model = gp.Model("p_median_balance")
    X = model.addVars(L, vtype=GRB.BINARY, name="X")
    Y = model.addVars(E, L, vtype=GRB.BINARY, name="Y")
    if alpha == 1:
        t = model.addVars(L, lb=0.0, name="t")

    # exactly p open facilities
    model.addConstr(X.sum() == p, name="num_facilities")

    # each demand covered once; no assignment to closed sites
    for i in E:
        model.addConstr(Y.sum(i, "*") == 1, name=f"assign_{i}")
        for j in L:
            model.addConstr(Y[i, j] <= X[j], name=f"link_{i}_{j}")

    # balancing term definition
    if alpha == 1:
        for j in L:
            model.addConstr(t[j] == gp.quicksum(a[i] * Y[i, j] for i in E),
                            name=f"t_def_{j}")

    # objective
    if alpha == 0:
        obj = gp.quicksum(a[i] * d[i, j] * Y[i, j] for i in E for j in L)
        model.setObjective(obj, GRB.MINIMIZE)
    else:
        expr = gp.QuadExpr()
        for i in E:
            for j in L:
                expr.add(a[i] * d[i, j] * Y[i, j] * (t[j] / A))
        model.setObjective(expr, GRB.MINIMIZE)
        model.Params.NonConvex = 2

    model.optimize()
    return model, X, Y


def add_p_via_mip(E, L, a, d, X_keep, p_add, alpha=0):
    """
    Starting from an existing set X_keep that stays open,
    open p_add additional facilities (exactly).
    """
    p_new = len(X_keep) + p_add
    A = sum(a[i] for i in E)

    model = gp.Model("p_median_add_mip")
    X = model.addVars(L, vtype=GRB.BINARY, name="X")
    Y = model.addVars(E, L, vtype=GRB.BINARY, name="Y")
    if alpha == 1:
        t = model.addVars(L, lb=0.0, name="t")

    # fix existing keeps
    for j in X_keep:
        model.addConstr(X[j] == 1, name=f"fix_keep_{j}")

    # open exactly p_new
    model.addConstr(X.sum() == p_new, name="num_facilities")

    # assignment constraints
    for i in E:
        model.addConstr(Y.sum(i, "*") == 1, name=f"assign_{i}")
        for j in L:
            model.addConstr(Y[i, j] <= X[j], name=f"link_{i}_{j}")

    # balancing term
    if alpha == 1:
        for j in L:
            model.addConstr(t[j] == gp.quicksum(a[i] * Y[i, j] for i in E),
                            name=f"t_def_{j}")

    # objective
    if alpha == 0:
        obj = gp.quicksum(a[i] * d[i, j] * Y[i, j] for i in E for j in L)
        model.setObjective(obj, GRB.MINIMIZE)
    else:
        expr = gp.QuadExpr()
        for i in E:
            for j in L:
                expr.add(a[i] * d[i, j] * Y[i, j] * (t[j] / A))
        model.setObjective(expr, GRB.MINIMIZE)
        model.Params.NonConvex = 2

    model.optimize()
    return model, X, Y


def greedy_add(E, L, a, d, X_keep, p_add, alpha=0):
    """
    Greedy heuristic: start from X_keep, add p_add sites one-by-one,
    each time picking the site that most reduces the objective.
    Returns the final list of open sites.
    """
    def cost_of(Xset):
        # assign each demand to nearest in Xset, apply balancing if alpha>0
        A = sum(a[i] for i in E)
        cover = {j: 0.0 for j in Xset}
        total = 0.0
        for i in E:
            j0 = min(Xset, key=lambda j: d[i, j])
            total += a[i] * d[i, j0]
            cover[j0] += a[i]
        if alpha > 0:
            bal = {j: (cover[j] / A) ** alpha for j in Xset}
            total = sum(a[i] * d[i, min(Xset, key=lambda j: d[i, j])] * \
                        bal[min(Xset, key=lambda j: d[i, j])]
                        for i in E)
        return total

    current = list(X_keep)
    for _ in range(p_add):
        best_j, best_cost = None, float('inf')
        for j in L:
            if j in current:
                continue
            trial = current + [j]
            c = cost_of(trial)
            if c < best_cost:
                best_cost, best_j = c, j
        current.append(best_j)
    return current


def relocate_via_mip(E, L, a, d, X_exist, M, alpha=0):
    """
    Relocate p=len(M) stations: close those in M, keep the rest,
    then open replacements via exact MIP.
    """
    X_keep = [j for j in X_exist if j not in M]
    return add_p_via_mip(E, L, a, d, X_keep, p_add=len(M), alpha=alpha)


def relocate_via_greedy(E, L, a, d, X_exist, M, alpha=0):
    """
    Relocate p=len(M) stations via greedy heuristic: close M,
    then greedy-add p replacements.
    """
    X_keep = [j for j in X_exist if j not in M]
    return greedy_add(E, L, a, d, X_keep, p_add=len(M), alpha=alpha)


