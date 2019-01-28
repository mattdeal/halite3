#!/usr/bin/env python3
# Python 3.6

# Import the Halite SDK, which will let you interact with the game.
import math
import logging
import numpy as np
import hlt
from hlt import constants
from hlt.positionals import Direction
from hlt.positionals import Position


class Bot16:

    def __init__(self, game):
        self.game = game

    def run(self):
        """ <<<Game Begin>>> """

        # This game object contains the initial game state.
        # game = hlt.Game()
        game = self.game

        # Give ships extra time to make it back to a dock, may not be necessary with drop offs in place now
        TURN_REMAINING_FUDGE = 6
        # Don't go back out if the game is over, sit and get crashed
        # TURN_NO_NEW_ORDERS = 16
        # Build new ships until this turn
        BUILD_UNTIL = 200
        # Maximum number of drop offs to build
        MAX_DROPOFF = 2
        # Drop offs must be built at least this far from any other dock
        MIN_DIST_TO_DOCK = 16
        # width and height of a square used to find halite clusters
        CLUSTER_SIZE = 2
        # width and height of the game map
        MAP_SIZE = game.game_map.width
        # maximum number of turns in this game
        MAX_TURNS = 400
        # the number of players in the game
        NUM_PLAYERS = len(game.players)
        # which job is assigned to this ship
        SHIP_JOBS = {}
        # which position this ship is trying to reach
        SHIP_ORDERS = {}
        # amount of halite to return with
        RETURN_WITH = 900

        # todo: tune the following params

        # distance at which a ship counts towards NEAREST_SHIP_COUNT
        NEAREST_SHIP_DIST = 10
        # number of ships that must be nearby to consider making a dock
        NEAREST_SHIP_COUNT = 5
        # amount of halite gained required to move to a new tile
        REQUIRED_GAIN = 1.4
        # amount of gain required during CLEAN before switching to CLUSTER
        BEST_GAIN_THRESHOLD = 5
        # todo: determine if this param is valuable, block tiles near enemy ships in 4p games
        ENEMY_RADIUS = 0

        # set default params per map size
        if MAP_SIZE == 40:
            MAX_TURNS = 425
            BUILD_UNTIL = 225
            MAX_DROPOFF = 2
            # MIN_DIST_TO_DOCK = 18
            # TURN_REMAINING_FUDGE = 6
            # TURN_NO_NEW_ORDERS = 14
            CLUSTER_SIZE = 2
        elif MAP_SIZE == 48:
            MAX_TURNS = 450
            BUILD_UNTIL = 250
            MAX_DROPOFF = 3
            # MIN_DIST_TO_DOCK = 20
            # TURN_REMAINING_FUDGE = 6
            # TURN_NO_NEW_ORDERS = 10
            CLUSTER_SIZE = 2
        elif MAP_SIZE == 56:
            MAX_TURNS = 475
            BUILD_UNTIL = 275
            MAX_DROPOFF = 4
            # MIN_DIST_TO_DOCK = 22
            # TURN_REMAINING_FUDGE = 6
            # TURN_NO_NEW_ORDERS = 8
            CLUSTER_SIZE = 2
        elif MAP_SIZE == 64:
            MAX_TURNS = 500
            BUILD_UNTIL = 300
            MAX_DROPOFF = 6
            # MIN_DIST_TO_DOCK = 24
            # TURN_REMAINING_FUDGE = 6
            # TURN_NO_NEW_ORDERS = 6
            CLUSTER_SIZE = 2

        # update params based on the number of players
        if NUM_PLAYERS > 2:
            MAX_DROPOFF -= 1
            ENEMY_RADIUS = 1

        # calculate clusters
        cluster_values_o = [[0 for x in range(MAP_SIZE)] for y in range(MAP_SIZE)]
        for x in range(MAP_SIZE):
            for y in range(MAP_SIZE):
                for y1 in range(y - CLUSTER_SIZE, y + CLUSTER_SIZE):
                    for x1 in range(x - CLUSTER_SIZE, x + CLUSTER_SIZE):
                        cluster_values_o[y][x] += game.game_map[Position(x1, y1)].halite_amount

        cluster_avg_o = np.matrix(cluster_values_o).mean()
        cluster_std_o = np.matrix(cluster_values_o).std()

        game.ready("v19-2p")

        """ <<<Game Loop>>> """

        while True:
            game.update_frame()
            game_map = game.game_map
            me = game.me
            halite_available = me.halite_amount
            command_queue = []
            ship_moves = {}

            # Keep up with how many turns are left
            turns_remaining = MAX_TURNS - game.turn_number
            logging.info("turns remaining {}".format(turns_remaining))

            # calculate current cluster values
            cluster_values_n = [[0 for x in range(MAP_SIZE)] for y in range(MAP_SIZE)]
            tile_values_n = [[0 for x in range(MAP_SIZE)] for y in range(MAP_SIZE)]
            for x in range(MAP_SIZE):
                for y in range(MAP_SIZE):
                    tile_values_n[y][x] = game.game_map[Position(x, y)].halite_amount
                    for y1 in range(y - CLUSTER_SIZE, y + CLUSTER_SIZE):
                        for x1 in range(x - CLUSTER_SIZE, x + CLUSTER_SIZE):
                            cluster_values_n[y][x] += game.game_map[Position(x1, y1)].halite_amount

            cluster_avg_n = np.matrix(cluster_values_n).mean()
            cluster_std_n = np.matrix(cluster_values_n).std()
            cluster_threshold = cluster_avg_n + (cluster_std_n * .5)

            tiles_avg_n = np.matrix(tile_values_n).mean()
            tiles_std_n = np.matrix(tile_values_n).std()

            # dropoff calculations
            dropoffs = me.get_dropoffs()
            dropoff_p1 = halite_available > constants.DROPOFF_COST and len(dropoffs) < MAX_DROPOFF
            dropoff_built = False

            # calculate blocked/dangerous tiles
            # permanently blocked cells - enemy shipyards, update with enemy docks as game progresses
            blocked_tiles_n = [[False for x in range(MAP_SIZE)] for y in range(MAP_SIZE)]
            danger_tiles_n = [[0 for x in range(MAP_SIZE)] for y in range(MAP_SIZE)]
            for player in game.players.values():
                if player.id == game.me.id:
                    continue

                blocked = player.shipyard.position

                blocked_tiles_n[blocked.y][blocked.x] = True

            for player in game.players.values():
                if player.id == me.id:
                    continue

                for ship in player.get_ships():
                    blocked_tiles_n[ship.position.y][ship.position.x] = True

                    # in 4p games, consider all tiles around enemies dangerous
                    if ENEMY_RADIUS > 0:
                        for p_blocked in ship.position.get_surrounding_cardinals():
                            p_norm = game_map.normalize(p_blocked)
                            danger_tiles_n[p_norm.y][p_norm.x] += 1

                for dropoff in player.get_dropoffs():
                    blocked_tiles_n[dropoff.position.y][dropoff.position.x] = True

                    # always consider the tiles around a dropoff extremely dangerous
                    for x in range(dropoff.position.x - 1, dropoff.position.x + 1):
                        for y in range(dropoff.position.y - 1, dropoff.position.y + 1):
                            p_norm = game_map.normalize(Position(x, y))
                            danger_tiles_n[p_norm.y][p_norm.x] += 4
                            # blocked_tiles_n[p_norm.y][p_norm.x] = True

            # clear my shipyard
            blocked_tiles_n[me.shipyard.position.y][me.shipyard.position.x] = False

            # clear my dropoffs
            for dropoff in dropoffs:
                blocked_tiles_n[dropoff.position.y][dropoff.position.x] = False

            # iterate through my ships and block their locations
            # this is done outside of the all players loop so that I ignore enemy ships in my docks
            for ship in me.get_ships():
                blocked_tiles_n[ship.position.y][ship.position.x] = True

            # clean up ship orders by removing orders tied to ships that have been lost
            orders_to_clear = []
            for key in SHIP_ORDERS.keys():
                try:
                    if me.get_ship(key) is None:
                        orders_to_clear.append(key)
                except KeyError:
                    orders_to_clear.append(key)

            for key in orders_to_clear:
                del SHIP_ORDERS[key]

            # iterate through my ships and give them orders
            for ship in me.get_ships():
                logging.info("Ship {} at position {} has {} halite.".format(ship.id, ship.position, ship.halite_amount))
                logging.info("Ship {} job {} target {}".format(ship.id, SHIP_JOBS.get(ship.id), SHIP_ORDERS.get(ship.id)))

                # find closest dock, starting with shipyard
                closest_dock = me.shipyard
                distance_to_dock = game_map.calculate_distance(ship.position, me.shipyard.position)

                # assign default job to new ships and order them out of the shipyard
                if SHIP_JOBS.get(ship.id) is None or distance_to_dock == 0:
                    SHIP_JOBS[ship.id] = "CLEAN"
                    unsafe_moves = Direction.get_all_cardinals()
                    logging.info("Ship {} needs to gtfo of the dock".format(ship.id, len(unsafe_moves)))
                    logging.info("Ship {} submitted {} potential moves".format(ship.id, len(unsafe_moves)))
                    ship_moves[ship.id] = unsafe_moves

                    # on to the next ship
                    continue

                # check for closer dropoff
                for dropoff in dropoffs:
                    new_dist = game_map.calculate_distance(ship.position, dropoff.position)
                    if new_dist < distance_to_dock:
                        distance_to_dock = new_dist
                        closest_dock = dropoff

                logging.info("Ship {} closest_dock {} distance {}".format(ship.id, closest_dock, distance_to_dock))

                # determine if this ship should convert to a dropoff
                if dropoff_p1 and not dropoff_built and distance_to_dock > MIN_DIST_TO_DOCK:
                    logging.info('todo: determine if ship should become dropoff')
                    dropoff_p2 = False
                    dropoff_p3 = False

                    # evaluate cluster value at ship position
                    if cluster_values_n[ship.position.y][ship.position.x] > cluster_threshold:
                        dropoff_p2 = True

                    if dropoff_p2:
                        nearby_ships = []

                        for ship_a in me.get_ships():
                            # ignore this ship
                            if ship_a.id == ship.id:
                                continue

                            if game_map.calculate_distance(ship_a.position, ship.position) < NEAREST_SHIP_DIST:
                                nearby_ships.append(ship_a)

                            if len(nearby_ships) > NEAREST_SHIP_COUNT:
                                dropoff_p3 = True
                                break

                        if dropoff_p3:
                            logging.info('should become dropoff')
                            dropoff_built = True
                            halite_available -= constants.DROPOFF_COST
                            command_queue.append(ship.make_dropoff())
                            blocked_tiles_n[ship.position.y][ship.position.x] = False

                            # move on to next ship
                            continue

                # start heading home if the game is almost over
                if turns_remaining - TURN_REMAINING_FUDGE <= distance_to_dock:
                    logging.info("Ship {} converted to {}".format(ship.id, 'CRASH'))
                    SHIP_JOBS[ship.id] = "CRASH"

                # Job == CRASH
                if SHIP_JOBS.get(ship.id) == "CRASH":
                    logging.info("Ship {} job {}".format(ship.id, 'CRASH'))

                    # do nothing, game is almost over
                    if distance_to_dock == 0:
                        logging.info("Ship {} waiting for the end".format(ship.id, 'RETURN'))

                        continue

                    # get moves that will move this ship towards the closest dock
                    unsafe_moves = game_map.get_unsafe_moves(ship.position, closest_dock.position)

                    # move into dock, ignoring collisions
                    if distance_to_dock == 1:
                        logging.info("Ship {} crashing dock {}".format(ship.id, closest_dock))

                        for move in unsafe_moves:
                            command_queue.append(ship.move(move))
                            blocked_tiles_n[ship.position.y][ship.position.x] = False

                            break

                        # next ship
                        continue
                    else:
                        logging.info("Ship {} submitted {} potential moves".format(ship.id, len(unsafe_moves)))
                        ship_moves[ship.id] = unsafe_moves

                # return home if carrying a lot of halite
                if ship.halite_amount > RETURN_WITH:
                    SHIP_JOBS[ship.id] = "RETURN"

                # JOB == RETURN
                if SHIP_JOBS.get(ship.id) == "RETURN":
                    logging.info("Ship {} job {}".format(ship.id, 'RETURN'))

                    # in the dock already
                    if distance_to_dock == 0:
                        # game almost over
                        if "CRASH" in SHIP_JOBS.values():
                            logging.info("Ship {} waiting for the end".format(ship.id))

                            # todo: make sure I don't need to mark this as free

                            # wait to be destroyed
                            continue

                        # assign new job
                        logging.info("Ship {} assigned job {}".format(ship.id, "CLEAN"))
                        SHIP_JOBS[ship.id] = "CLEAN"

                        # get out of the dock
                        # del SHIP_JOBS[ship.id]
                        # unsafe_moves = Direction.get_all_cardinals()
                        # logging.info("Ship {} needs to gtfo of the dock".format(ship.id, len(unsafe_moves)))
                        # logging.info("Ship {} submitted {} potential moves".format(ship.id, len(unsafe_moves)))
                        # ship_moves[ship.id] = unsafe_moves
                    elif distance_to_dock == 1:
                        if "CRASH" in SHIP_JOBS.values():
                            logging.info("Ship {} crashing dock {}".format(ship.id, closest_dock))

                            unsafe_moves = game_map.get_unsafe_moves(ship.position, closest_dock.position)
                            for move in unsafe_moves:
                                command_queue.append(ship.move(move))
                                blocked_tiles_n[ship.position.y][ship.position.x] = False

                                break

                            # move to next ship
                            continue
                        else:
                            unsafe_moves = game_map.get_unsafe_moves(ship.position, closest_dock.position)
                            logging.info("Ship {} submitted {} potential moves".format(ship.id, len(unsafe_moves)))
                            ship_moves[ship.id] = unsafe_moves
                    else:
                        logging.info("Ship {} heading to dock {}".format(ship.id, closest_dock))
                        unsafe_moves = game_map.get_unsafe_moves(ship.position, closest_dock.position)
                        logging.info("Ship {} submitted {} potential moves".format(ship.id, len(unsafe_moves)))
                        ship_moves[ship.id] = unsafe_moves

                # JOB == CLEAN
                if SHIP_JOBS.get(ship.id) == "CLEAN":
                    logging.info("Ship {} job {}".format(ship.id, 'CLEAN'))

                    cur_pos_val = game_map[ship.position].halite_amount
                    cur_pos_gain = math.ceil(cur_pos_val * .25)
                    cur_pos_val_next = cur_pos_val - cur_pos_gain
                    cost_to_move = math.floor(cur_pos_val / 10)
                    nearby_positions = ship.position.get_surrounding_cardinals()
                    best_gain = cur_pos_gain
                    best_position = ship.position
                    min_gain = cur_pos_gain * REQUIRED_GAIN
                    ranked_positions = {}

                    for position in nearby_positions:
                        p_norm = game_map.normalize(position)

                        logging.info("Ship {} considering {}.".format(ship.id, p_norm))

                        # ignore closest dock
                        if p_norm == closest_dock.position:
                            continue

                        tile_value = game_map[p_norm].halite_amount
                        potential_gain = math.ceil(tile_value * .25) - cost_to_move

                        logging.info("Ship {} considering {} halite {}.".format(ship.id, p_norm, tile_value))

                        if min_gain < potential_gain:
                            logging.info("Ship {} considering {} potential_gain {}.".format(ship.id, p_norm, potential_gain))
                            ranked_positions[potential_gain] = p_norm

                            if potential_gain > best_gain:
                                best_position = p_norm
                                best_gain = potential_gain

                        # todo: confirm this works, sometimes we might be better off sitting still

                    if best_gain < BEST_GAIN_THRESHOLD:
                        logging.info("Ship {} assigned job {}".format(ship.id, "CLUSTER"))
                        SHIP_JOBS[ship.id] = "CLUSTER"
                    elif best_gain == cur_pos_gain:
                        logging.info("Ship {} {}".format(ship.id, "staying still"))
                    else:
                        # order positions and append to requested moves
                        unsafe_moves = []
                        for k in sorted(ranked_positions.keys(), reverse=True):
                            unsafe_moves += game_map.get_unsafe_moves(ship.position, ranked_positions[k])

                        logging.info("Ship {} submitted {} potential moves".format(ship.id, len(unsafe_moves)))
                        ship_moves[ship.id] = unsafe_moves

                # JOB == CLUSTER
                if SHIP_JOBS.get(ship.id) == "CLUSTER":
                    logging.info("Ship {} job {}".format(ship.id, 'CLUSTER'))
                    needs_assignment = True

                    # validate existing order
                    if SHIP_ORDERS.get(ship.id):
                        order = SHIP_ORDERS.get(ship.id)
                        logging.info("Ship {} has existing assignment {}".format(ship.id, order))

                        # if ship at target, clean, else determine if target is still valid
                        if game_map.calculate_distance(ship.position, order) == 0:
                            logging.info("Ship {} reached target {}".format(ship.id, order))
                            logging.info("Ship {} {}".format(ship.id, "staying still"))
                            logging.info("Ship {} assigned job {}".format(ship.id, "CLEAN"))

                            del SHIP_ORDERS[ship.id]
                            SHIP_JOBS[ship.id] = "CLEAN"
                            needs_assignment = False
                        else:
                            logging.info("Ship {} has not reached target {}".format(ship.id, order))
                            cluster_ratio = cluster_values_n[order.y][order.x] / cluster_values_o[order.y][order.x]
                            if cluster_ratio < .6:
                                # todo: this might not work towards the end of a game with high performance bots
                                logging.info("Ship {} target {} remaining {} - new target".format(ship.id, order, cluster_ratio))
                            elif game_map[order].is_occupied:
                                logging.info("Ship {} target {} occupied - new target".format(ship.id, order))
                            else:
                                needs_assignment = False
                                logging.info('Ship {} continuing towards {}'.format(ship.id, order))
                                unsafe_moves = game_map.get_unsafe_moves(ship.position, order)
                                logging.info("Ship {} submitted {} potential moves".format(ship.id, len(unsafe_moves)))
                                ship_moves[ship.id] = unsafe_moves

                    if needs_assignment:
                        logging.info("Ship {} {}".format(ship.id, 'getting new assignment'))

                        clusters = {}
                        for x in range(MAP_SIZE):
                            for y in range(MAP_SIZE):
                                cluster_value = cluster_values_n[y][x]
                                if cluster_value > cluster_threshold:
                                    cluster_position = Position(x, y)
                                    cluster_distance = game_map.calculate_distance(ship.position, cluster_position)

                                    if cluster_distance == 0:
                                        cluster_distance = 1
                                    else:
                                        cluster_distance = cluster_distance * cluster_distance

                                    clusters[cluster_value / cluster_distance] = cluster_position

                        if len(clusters.keys()) > 0:
                            best_position = None

                            for cluster in sorted(clusters.keys(), reverse=True):
                                logging.info('Ship {} testing Position {} value {}'.format(ship.id, clusters[cluster], cluster))

                                if game_map[clusters[cluster]].is_occupied:
                                    logging.info('Ship {} Position {} is occupied'.format(ship.id, clusters[cluster]))
                                    continue

                                if clusters[cluster] in SHIP_ORDERS.values():
                                    logging.info('Ship {} Position {} is already assigned'.format(ship.id, clusters[cluster]))
                                    continue

                                # determine if there are too many ships close to this cluster already
                                x = clusters[cluster].x
                                y = clusters[cluster].y
                                ships_in_cluster = 0
                                for y1 in range(y - CLUSTER_SIZE, y + CLUSTER_SIZE):
                                    for x1 in range(x - CLUSTER_SIZE, x + CLUSTER_SIZE):
                                        if game_map[Position(x1, y1)].is_occupied:
                                            ships_in_cluster += 1

                                # todo: validate this number is relevant
                                if ships_in_cluster > 10:
                                    continue

                                # todo: determine if this ship can move towards cluster

                                best_position = clusters[cluster]
                                SHIP_ORDERS[ship.id] = best_position
                                break

                            if best_position:
                                logging.info('Ship {} targeting best cluster {}'.format(ship.id, best_position))
                                unsafe_moves = game_map.get_unsafe_moves(ship.position, best_position)
                                logging.info("Ship {} submitted {} potential moves".format(ship.id, len(unsafe_moves)))
                                ship_moves[ship.id] = unsafe_moves

            # attempt to solve traffic jams
            changes_made = True
            while changes_made:
                if len(ship_moves.keys()) == 0:
                    break

                changes_made = False
                ship_ids_resolved = []

                for k in ship_moves.keys():
                    ship = me.get_ship(k)
                    unsafe_moves = ship_moves[k]

                    # if ship does not have enough halite to move, mark it resolved
                    cur_pos_val = game_map[ship.position].halite_amount
                    cost_to_move = math.floor(cur_pos_val / 10)

                    if ship.halite_amount < cost_to_move:
                        logging.info('Ship {} cannot move'.format(ship.id))
                        ship_ids_resolved.append(k)
                        continue

                    for move in unsafe_moves:
                        potential_position = game_map.normalize(ship.position.directional_offset(move))
                        logging.info("Ship {} testing {}".format(ship.id, potential_position))

                        # tile is blocked
                        if blocked_tiles_n[potential_position.y][potential_position.x]:
                            logging.info("Ship {} testing {} is blocked".format(ship.id, potential_position))
                            logging.info("{} occupied {}".format(potential_position, game_map[potential_position].is_occupied))
                            continue

                        # tile is dangerous
                        if ENEMY_RADIUS > 0 and SHIP_JOBS.get(ship.id) == "CLEAN":
                            danger_rating = danger_tiles_n[potential_position.y][potential_position.x]

                            # ignore dangerous tiles
                            if danger_rating > 3:
                                logging.info("{} danger {}".format(potential_position, danger_rating))
                                continue

                            # todo: test danger ratio of tile if job is clean
                            if danger_rating > 0 and tile_values_n[potential_position.y][potential_position.x] < tiles_avg_n + (tiles_std_n * danger_rating):
                                continue

                        logging.info('Ship {} moving to {}'.format(ship.id, potential_position))
                        command_queue.append(ship.move(move))
                        blocked_tiles_n[ship.position.y][ship.position.x] = False
                        blocked_tiles_n[potential_position.y][potential_position.x] = True
                        changes_made = True
                        ship_ids_resolved.append(k)
                        break

                # clear out resolved moves
                if len(ship_ids_resolved) > 0:
                    logging.info('Resolved {} ships'.format(len(ship_ids_resolved)))
                    for ship_id in ship_ids_resolved:
                        del ship_moves[ship_id]

            # ships could not move due to blocked tiles, attempt to swap
            if len(ship_moves.keys()) > 0:
                logging.info("{} ships did not move, attempting swaps".format(len(ship_moves.keys())))

                changes_made = True
                while changes_made:
                    if len(ship_moves.keys()) == 0:
                        break

                    changes_made = False
                    ship_ids_resolved = []

                    for k in ship_moves.keys():
                        for k1 in ship_moves.keys():
                            if k == k1:
                                continue

                            ship_a = me.get_ship(k)
                            ship_b = me.get_ship(k1)

                            if game_map.calculate_distance(ship_a.position, ship_b.position) > 1:
                                continue

                            positions_a = []
                            moves_a = ship_moves[k]
                            for move in moves_a:
                                positions_a.append(game_map.normalize(ship_a.position.directional_offset(move)))

                            positions_b = []
                            moves_b = ship_moves[k1]
                            for move in moves_b:
                                positions_b.append(game_map.normalize(ship_b.position.directional_offset(move)))

                            if ship_a.position in positions_b and ship_b.position in positions_a:
                                logging.info("ship {} at {} can swap with ship {} at {}".format(k, ship_a.position, k1, ship_b.position))

                                unsafe_a = game_map.get_unsafe_moves(ship_a.position, ship_b.position)
                                unsafe_b = game_map.get_unsafe_moves(ship_b.position, ship_a.position)

                                command_queue.append(ship_a.move(unsafe_a[0]))
                                command_queue.append(ship_b.move(unsafe_b[0]))

                                changes_made = True
                                ship_ids_resolved.append(k)
                                ship_ids_resolved.append(k1)
                                break

                        if changes_made:
                            # ship a and b swapped, neither one is valid anymore and they need to be cleaned up
                            break

                    # clear out resolved moves
                    if len(ship_ids_resolved) > 0:
                        logging.info("Swapping resolved {} ships".format(len(ship_ids_resolved)))
                        for ship_id in ship_ids_resolved:
                            del ship_moves[ship_id]

            logging.info("{} ships did not move".format(len(ship_moves.keys())))

            # spawn ships
            if game.turn_number <= BUILD_UNTIL and halite_available >= constants.SHIP_COST and not \
                    blocked_tiles_n[me.shipyard.position.y][me.shipyard.position.x]:
                command_queue.append(me.shipyard.spawn())
                halite_available -= constants.SHIP_COST

            # Send your moves back to the game environment, ending this turn.
            game.end_turn(command_queue)

